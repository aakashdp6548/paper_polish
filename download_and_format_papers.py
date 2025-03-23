import os
import json
import argparse
import requests
from tqdm import tqdm
from datasets import Dataset
import PyPDF2
from concurrent.futures import ThreadPoolExecutor
import tempfile

def download_pdf(paper_id, save_dir):
    """Download a PDF from OpenReview and save it locally."""
    url = f"https://openreview.net/pdf?id={paper_id}"
    pdf_path = os.path.join(save_dir, f"{paper_id}.pdf")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return pdf_path
    except Exception as e:
        print(f"Error downloading PDF for {paper_id}: {e}")
        return None

def extract_text_from_pdf(pdf_path, page_limit):
    """Extract plain text from a PDF file."""
    if not pdf_path or not os.path.exists(pdf_path):
        return ""

    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            # Read up to first page_limit pages
            for page_num in range(min(len(pdf_reader.pages), page_limit)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def process_paper(paper, save_dir, page_limit):
    """Process a single paper: download PDF and extract text."""
    paper_id = paper["id"]
    pdf_path = download_pdf(paper_id, save_dir)
    paper_text = extract_text_from_pdf(pdf_path, page_limit)
    
    # Remove PDF file after extraction to save space
    if pdf_path and os.path.exists(pdf_path):
        os.remove(pdf_path)
    
    # Create the dataset entry
    entry = {
        "model_input": paper_text,
        "result": paper["status"].lower()
    }

    # Add all other metadata fields
    for key, value in paper.items():
        if key not in entry:
            entry[key] = value
    
    return entry

def create_dataset(papers, output_path, num_workers=4, page_limit=10):
    """Create a HuggingFace dataset from papers."""
    print(f"Processing {len(papers)} papers...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process papers in parallel
        entries = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_paper, paper, temp_dir, page_limit) for paper in papers]
            for future in tqdm(futures, total=len(papers)):
                entry = future.result()
                if entry["model_input"]:  # Only add entries with text
                    entries.append(entry)
        
        # Create dataset
        dataset = Dataset.from_list(entries)
        
        # Save dataset
        dataset.save_to_disk(output_path)
        print(f"Dataset with {len(dataset)} entries saved to {output_path}")
    
    # Return some statistics
    return {
        "total_papers": len(papers),
        "processed_papers": len(dataset),
        "skipped_papers": len(papers) - len(dataset)
    }

def main():
    parser = argparse.ArgumentParser(description="Create a HuggingFace dataset from OpenReview papers")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for testing (optional)")
    parser.add_argument("--page_limit", type=int, default=10, help="Number of pages to read from the PDF")
    args = parser.parse_args()
    
    # Load papers from JSON
    with open(args.input_json, 'r') as f:
        papers = json.load(f)
    
    # Take a sample if specified
    if args.sample_size:
        import random
        random.seed(42)
        papers = random.sample(papers, min(args.sample_size, len(papers)))
    
    # Create dataset
    stats = create_dataset(papers, args.output_path, args.num_workers)
    print(f"Processing complete. Statistics: {stats}")

if __name__ == "__main__":
    main()