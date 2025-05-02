import os
from openai import OpenAI
from datasets import load_from_disk

os.environ["OPENAI_API_KEY"] = "..."

# load paper data
papers = load_from_disk("/gpfs/radev/home/tj372/project/paper_polish/merged_paper_reviews_2025")
PAPER_TEXT = papers["train"]["model_input"][7594] # 7594 is index of specific paper to analyze


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

model_instructions = """You are an expert machine learning researcher helping to improve a paper for submission to ICLR.
Given the research paper below, suggest concrete and actionable improvements that address the paper's weak points while leveraging its strengths.
"""

# prompt the model to generate improvements
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": model_instructions},
        {"content": PAPER_TEXT, "role": "user"},
    ],
    max_tokens=15000,
)
output_text = completion.choices[0].message.content


os.makedirs("model_outputs", exist_ok=True)
with open("model_outputs/single_agent_output.txt", "w") as f:
    f.write("FINAL OUTPUT:\n")
    f.write(output_text + "\n")

print(output_text)


