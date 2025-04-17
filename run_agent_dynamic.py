import os
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.llms import HuggingFacePipeline

# Set API key 
os.environ["OPENAI_API_KEY"] = "..."

# debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-bacb3cb8-ea7a-550f-9421-3cc940ebd0c8"


# ---------- Initialize LLMs ----------

# Supervisor agent using GPT for synthesizing improvements.
supervisor_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=10000,
    timeout=None,
    max_retries=2,
)

# Reviewer agents using Gemma-3-4b-it for analyzing strengths and weaknesses.
model_name = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    print("Warning: Tokenizer does not have a pad token. Setting pad_token = eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500, temperature=0.2)

reviewer_llm = HuggingFacePipeline(pipeline=pipe)


# ---------- Define the individual chains ----------

# Strengths Chain: Given a paper text, output its strengths.
strengths_prompt_template = """
<bos><start_of_turn>user
### INSTRUCTIONS:
You are an expert machine learning researcher. Given the content of a paper submitted to the International Conference on Learning Representations (ICLR)
write a helpful review that highlights the paper's strengths.
### PAPER:
{paper}

Generate only the strengths of the paper.
<end_of_turn>

<start_of_turn>model
"""
strengths_prompt = PromptTemplate(
    input_variables=["paper"], template=strengths_prompt_template
)
strengths_chain = strengths_prompt | reviewer_llm | StrOutputParser()


weaknesses_prompt_template = """
<bos><start_of_turn>user
### INSTRUCTIONS:
You are an expert machine learning researcher. Given the content of a paper submitted to the International Conference on Learning Representations (ICLR)
write a helpful review that highlights the paper's weaknesses.
### PAPER:
{paper}

Generate only the weaknesses of the paper.
<end_of_turn>

<start_of_turn>model
"""
weaknesses_prompt = PromptTemplate(
    input_variables=["paper"], template=weaknesses_prompt_template
)
weaknesses_chain = weaknesses_prompt | reviewer_llm | StrOutputParser()



# Synthesis Chain: Use the paper text along with its strengths and weaknesses to output actionable improvement suggestions.
synthesis_prompt_template = """
You are an expert machine learning researcher helping to improve a paper for submission to the International Conference on Learning Representations (ICLR).
Given the research paper below along with its identified strengths and weaknesses,
suggest concrete and actionable improvements that address the paper's weak points while leveraging its strengths.
Explain how the paper might be rewritten, how experiments could be enhanced,
and how various sections could be clarified to boost its chances of acceptance.

Paper:
{paper}

Strengths:
{strengths}

Weaknesses:
{weaknesses}

Improvements:"""

synthesis_prompt = PromptTemplate(
    input_variables=["paper", "strengths", "weaknesses"], template=synthesis_prompt_template
)
synthesis_chain = synthesis_prompt | supervisor_llm | StrOutputParser()



# ---------- Define the functions to be registered as tools. ----------
def strengths_tool(paper: str) -> str:
    """Tool to extract strengths from the paper text."""
    return strengths_chain.invoke({"paper": paper})

def weaknesses_tool(paper: str) -> str:
    """Tool to extract weaknesses from the paper text."""
    return weaknesses_chain.invoke({"paper": paper})

def synthesis_tool(paper: str) -> str:
    """
    Tool to synthesize improvement suggestions.
    This function calls both reviewer tools and passes their results to the synthesis chain.
    """
    strengths = strengths_tool(paper)
    weaknesses = weaknesses_tool(paper)
    return synthesis_chain.invoke({
        "paper": paper,
        "strengths": strengths,
        "weaknesses": weaknesses
        })


# ---------- Register these functions as tools for the agent. ----------
tools = [
    Tool(
        name="Strengths Analyzer",
        func=strengths_tool,
        description="Given a research paper, returns its key strengths."
    ),
    Tool(
        name="Weaknesses Analyzer",
        func=weaknesses_tool,
        description="Given a research paper, returns its weaknesses."
    ),
    Tool(
        name="Synthesize Improvements",
        func=synthesis_tool,
        description="Given a research paper, synthesizes actionable improvements by leveraging the paper's strengths and weaknesses."
    )
]


# ---------- Initialize a dynamic (supervisor) agent using LangChain's Zero-Shot ReAct. ----------

PAPER_TAGS = "\n\n<PAPER>\n{paper}\n</PAPER>"

supervisor_prompt_template = '''Answer the following questions as best you can.
You have access to the following tools:

{tools}

**When you call a tool, you MUST pass the entire contents found between
<PAPER></PAPER> tags verbatim in `Action Input`.  Do NOT summarise.**

Use the following format:
Question: the input question you must answer
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: the input to the action (-- the full paper text! --)
Observation: the result of the action
(… you may repeat Thought / Action / Observation up to 3 times …)
Thought: I now know the final answer
Final Answer: … your recommendations …

Begin!

Question: {input}
Thought:{agent_scratchpad}''' + PAPER_TAGS

supervisor_question = """ You are an expert machine learning researcher helping to improve a paper for submission to ICLR.
Given the tools available to you and the research paper, can you synthesize actionable improvement suggestions
to improve the paper's chances of acceptance?"""

prompt = PromptTemplate.from_template(supervisor_prompt_template)



agent = create_react_agent(
    llm=supervisor_llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=4,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)


# ---------- Function to generate a final review and actionable steps. ----------
def dynamic_review_and_improve(paper_text: str) -> str:
    """
    Use the dynamic supervisor agent to generate a comprehensive review and improvement suggestions.
    The agent will call the strengths and weaknesses tools as needed before synthesizing its final output.
    """
    final_response = agent_executor.invoke({
        "input": supervisor_question,
        "paper": paper_text
    })
    return final_response



if __name__ == '__main__':
    
    papers = load_from_disk("/gpfs/radev/home/tj372/project/paper_polish/merged_paper_reviews_2025")
    papers_train, papers_test, papers_validation = papers["train"], papers["test"], papers["val"]
    
    paper_text = papers_train['model_input'][0]
    
    # Generate the final review and suggestions.
    result = dynamic_review_and_improve(paper_text)
    print("Final Review and Improvement Suggestions:\n")
    print(result)

    with open("multi_agent_review_output.txt", "w") as f:
        f.write("FINAL OUTPUT:\n")
        f.write(result["output"] + "\n\n")
        f.write("INTERMEDIATE STEPS:\n")
        for i, (thought, action_result) in enumerate(result["intermediate_steps"], 1):
            f.write(f"Step {i}:\n  Thought: {thought}\n  Result: {action_result}\n\n")
