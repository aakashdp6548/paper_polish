import os
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline

os.environ["OPENAI_API_KEY"] = "..."
# debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-a5189f65-9660-5b03-8e8f-a63f7146324d"

# ---------- Initialize LLMs ----------

# Supervisor agent using GPT for synthesizing improvements. Short token context for debugging
supervisor_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=1000,
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
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    temperature=0.1,
    )

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
synthesis_prompt_template = """You are an expert machine learning researcher helping to improve a paper for submission to ICLR.
Given the research paper below along with its identified strengths and weaknesses,
suggest concrete and actionable improvements that address the paper's weak points while leveraging its strengths.

Paper:
{paper}

Improvements:"""


synthesis_prompt = PromptTemplate(
    input_variables=["paper", "strengths", "weaknesses"],
    template=synthesis_prompt_template,
)
synthesis_chain = synthesis_prompt | supervisor_llm | StrOutputParser()



# ---------- Define the functions to be registered as tools. ----------
def strengths_tool(paper: str) -> str:
    """Tool to extract strengths from the paper text."""
    print("==== STRENGTHS TOOL CALLED ====")
    result = strengths_chain.invoke({"paper": paper})
    print("==== STRENGTHS TOOL COMPLETED ====")
    return result

def weaknesses_tool(paper: str) -> str:
    """Tool to extract weaknesses from the paper text."""
    print("==== WEAKNESSES TOOL CALLED ====")
    result = weaknesses_chain.invoke({"paper": paper})
    print("==== WEAKNESSES TOOL COMPLETED ====")
    return result 


def synthesis_tool(paper: str) -> str:
    print("==== SYNTHESIS TOOL CALLED ====")
    result = synthesis_chain.invoke({
        "paper": paper
    })
    print("==== SYNTHESIS TOOL COMPLETED ====")
    return result




# Only the synthesis tool is registered as a tool.
tools = [
    Tool(
        name="synthesize_improvements",
        func=synthesis_tool,
        description=(
           "Given a paper, its strengths, and its weaknesses, "
           "synthesizes actionable improvements. "
           "Input must be a JSON-like object with keys 'paper','strengths','weaknesses'."
        ),
    )
]


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
)


system_prompt_template = """\
You have access to the tool:

  1. synthesize_improvements  
     - Input: <full paper text> 
     - Output: actionable, concrete improvement suggestions  

Call `synthesize_improvements` once.
"""



memory = InMemoryChatMessageHistory(session_id="test-session-1")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(
    model,
    tools,
    prompt
    )
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    return_intermediate_steps=True,
    )

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

papers = load_from_disk("/gpfs/radev/home/tj372/project/paper_polish/merged_paper_reviews_2025")
papers_train, papers_test, papers_validation = papers["train"], papers["test"], papers["val"]

PAPER_TEXT = papers_train['model_input'][0]


config = {"configurable": {"session_id": "test-session-1"}}
result = agent_with_chat_history.invoke(
        {"input": PAPER_TEXT}, config
    )
print(result["output"] + "\n\n\n")
print(result["intermediate_steps"])


with open("run_single_agent_output.txt", "w") as f:
    f.write("FINAL OUTPUT:\n")
    f.write(result["output"] + "\n\n")

    f.write("INTERMEDIATE STEPS:\n")
    for i, (agent_action, observation) in enumerate(result["intermediate_steps"], 1):
        f.write(f"Step {i}:\n")
        f.write(f"  Thought: {agent_action.log.strip()}\n")
        f.write(f"  Action Used: {agent_action.tool}\n")
        f.write(f"  Action Input: {agent_action.tool_input}\n")
        f.write(f"  Result: {observation.strip()}\n\n\n\n")



