import os
import json
import time
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


def parse_official_review_text(extracted_text):
    """
    Given the raw text from an Official Review, parse it into the sections we need,
    """

    headings = [
        "Summary:",
        "Soundness:",
        "Presentation:",
        "Contribution:",
        "Strengths:",
        "Weaknesses:",
        "Questions:",
        "Flag For Ethics Review:",
        "Rating:",
        "Confidence:",
        "Code Of Conduct:",
    ]

    single_review_parsed = {
        "summary": "",
        "soundness": "",
        "presentation": "",
        "contribution": "",
        "strengths": "",
        "weaknesses": "",
        "questions": "",
        "flag_for_ethics_review": "",
        "rating": "",
        "confidence": "",
        "code_of_conduct": "",
    }

    current_label = None
    lines = extracted_text.splitlines()

    for line in lines:
        line_stripped = line.strip()
        found_heading = None

        # check if this line begins with any recognized heading
        for heading in headings:
            if line_stripped.startswith(heading):
                found_heading = heading
                break

        if found_heading:
            current_label = found_heading
            after_colon = line_stripped.replace(found_heading, "").strip()
            dict_key = (
                found_heading.lower()
                .replace(":", "")
                .replace(" ", "_")
            )
            single_review_parsed[dict_key] = after_colon
        else:
            if current_label:
                dict_key = (
                    current_label.lower()
                    .replace(":", "")
                    .replace(" ", "_")
                )
                single_review_parsed[dict_key] += "\n" + line_stripped

    # clean up leading/trailing whitespace
    for k in single_review_parsed:
        single_review_parsed[k] = single_review_parsed[k].strip()

    # remove trailing "Add:\nPublic Comment" from code_of_conduct
    coc_text = single_review_parsed["code_of_conduct"]
    pattern = r"(\s*Add:\s*Public Comment\s*)$"
    single_review_parsed["code_of_conduct"] = re.sub(pattern, "", coc_text).rstrip()

    return single_review_parsed


def scrape_forum_reviews(forum_id, driver):
    """
    Navigate to the OpenReview forum page for the given forum_id,
    scrape all 'Official Review' divs, parse them, and return a
    data structure in the format:
    
    {
      "id": forum_id,
      "summaries": [],
      "soundness": [],
      "presentation": [],
      "contribution": [],
      "strengths": [],
      "weaknesses": [],
      "questions": [],
      "flag_for_ethics_review": [],
      "rating": [],
      "confidence": [],
      "code_of_conduct": []
    }
    """
    forum_data = {
        "id": forum_id,
        "summaries": [],
        "soundness": [],
        "presentation": [],
        "contribution": [],
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "flag_for_ethics_review": [],
        "rating": [],
        "confidence": [],
        "code_of_conduct": []
    }

    url = f"https://openreview.net/forum?id={forum_id}"
    driver.get(url)
    time.sleep(1)  # wait a bit for the page to load

    # find divs that might contain Official Reviews
    divs = driver.find_elements(By.CSS_SELECTOR, "div.note.depth-odd")
    
    for div in divs:
        try:
            h4 = div.find_element(By.TAG_NAME, "h4")
            if "Official Review" in h4.text:

                parent_html = div.get_attribute("innerHTML")
                soup = BeautifulSoup(parent_html, "html.parser")

                # remove note replies if present
                note_replies = soup.find("div", class_="note-replies")
                if note_replies:
                    note_replies.decompose()

                extracted_text = soup.get_text(separator="\n", strip=True)

                # parse the extracted text into the structured dict
                single_review_parsed = parse_official_review_text(extracted_text)

                # accumulate into forum_data
                forum_data["summaries"].append(
                    single_review_parsed["summary"]
                )
                forum_data["soundness"].append(
                    single_review_parsed["soundness"]
                )
                forum_data["presentation"].append(
                    single_review_parsed["presentation"]
                )
                forum_data["contribution"].append(
                    single_review_parsed["contribution"]
                )
                forum_data["strengths"].append(
                    single_review_parsed["strengths"]
                )
                forum_data["weaknesses"].append(
                    single_review_parsed["weaknesses"]
                )
                forum_data["questions"].append(
                    single_review_parsed["questions"]
                )
                forum_data["flag_for_ethics_review"].append(
                    single_review_parsed["flag_for_ethics_review"]
                )
                forum_data["rating"].append(
                    single_review_parsed["rating"]
                )
                forum_data["confidence"].append(
                    single_review_parsed["confidence"]
                )
                forum_data["code_of_conduct"].append(
                    single_review_parsed["code_of_conduct"]
                )

        except Exception:
            pass

    return forum_data


def append_forum_data_to_json(forum_data, json_file="reviews.json"):
    """
    Loads existing data from `json_file` (if present and valid),
    appends the new `forum_data` dict,
    then saves back to JSON.
    """
    existing_data = []

    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing_data = [] #empty list if file is not valid JSON

    existing_data.append(forum_data)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)


def main():
    options = Options()
    options.add_argument("--headless")  
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # open iclr paper list
    with open('iclr2025.json', 'r') as f:
        data = json.load(f)

    try:
        for idx in tqdm(range(4762, len(data))):
            fid = data[idx]["id"]
            
            # scrape and parse forum reviews
            forum_data = scrape_forum_reviews(forum_id=fid, driver=driver)
            if forum_data["summaries"] == []:
                time.sleep(30)
                forum_data = scrape_forum_reviews(forum_id=fid, driver=driver)
            
            forum_data["title"] = data[idx]["title"]
            forum_data["track"] = data[idx]["track"]
            forum_data["status"] = data[idx]["status"]
            forum_data["keywords"] = data[idx]["keywords"]
            forum_data["corr_rating_confidence"] = data[idx]["corr_rating_confidence"]
            
            # append to json
            append_forum_data_to_json(forum_data, json_file="reviews.json")
            # print(f"Appended data for forum_id={fid} to reviews.json")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
