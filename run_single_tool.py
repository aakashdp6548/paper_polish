import os
from typing import Dict
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field


os.environ["OPENAI_API_KEY"] = "..."
# debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-6d25d029-31ac-5260-87fb-3e232ed96dad"


# ---------- Initialize LLMs ----------

# Supervisor agent using GPT for synthesizing improvements.
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

Strengths:
{strengths}

Weaknesses:
{weaknesses}

Improvements:"""

synthesis_prompt = PromptTemplate(
    input_variables=["paper", "strengths", "weaknesses"],
    template=synthesis_prompt_template,
)
synthesis_chain = synthesis_prompt | supervisor_llm | StrOutputParser()




@tool
def strengths_analyzer(paper: str) -> str:
    """Tool to extract strengths from the paper text."""
    print("==== STRENGTHS TOOL CALLED ====")
    result = strengths_chain.invoke({"paper": paper})
    print("==== STRENGTHS TOOL COMPLETED ====")
    return result

@tool
def weaknesses_analyzer(paper: str) -> str:
    """Tool to extract weaknesses from the paper text."""
    print("==== WEAKNESSES TOOL CALLED ====")
    result = weaknesses_chain.invoke({"paper": paper})
    print("==== WEAKNESSES TOOL COMPLETED ====")
    return result 


@tool
def synthesis_tool(paper: str) -> str:
    """
    Synthesizes the analysis of a paper based on its identified strengths and weaknesses.
    This tool MUST be called only AFTER analyzing strengths and weaknesses using the dedicated tools.
    """

    print("==== SYNTHESIS TOOL CALLED ====")

    strengths = strengths_chain.invoke({"paper": paper})
    weaknesses = weaknesses_chain.invoke({"paper": paper})

    
    print(f"Paper: {paper[:50]}...")
    print(f"Strengths: {strengths}")
    print(f"Weaknesses: {weaknesses}")

    print("\n\n")
    result = synthesis_chain.invoke({
        "paper": paper,
        "strengths": strengths,
        "weaknesses": weaknesses
    })

    print("==== SYNTHESIS TOOL COMPLETED ====")
    return result



tools = [
    Tool(
        name="synthesis_tool",
        func=synthesis_tool,
        description=(
            "Given a paper, synthesizes actionable improvements by leveraging past tool results. "
            "Will use the chat history to find strengths and weaknesses."
        )
    )
]

# --- History Management (Your existing code is fine) ---
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
    def clear(self) -> None:
        self.messages = []

store: Dict[str, BaseChatMessageHistory] = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]



# --- Prompt Engineering ---
# Crucially, instruct the LLM on the dependency!
system_prompt_template = """You are a helpful research assistant.
You have access to this tool for analyzing research papers.
    
    1. synthesis_tool
        - Input: the full paper text, no truncation or summarization.
        - Output: a list of actionable improvements based on the strengths and weaknesses.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        MessagesPlaceholder(variable_name="history"), # Populated by RunnableWithMessageHistory
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Intermediate steps (tool calls/outputs)
    ]
)

# --- Agent Setup ---
agent = create_tool_calling_agent(supervisor_llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5, 
    return_intermediate_steps=True,
)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_by_session_id,
    input_messages_key="input",
    history_messages_key="history",
)

# --- Running the Agent ---
session_id = "analysis_session_1"

papers = load_from_disk("/gpfs/radev/home/tj372/project/paper_polish/merged_paper_reviews_2025")
papers_train, papers_test, papers_validation = papers["train"], papers["test"], papers["val"]

PAPER_TEXT = papers_train['model_input'][0]

print("\n--- Invoking Agent for Full Analysis ---")
result = agent_with_chat_history.invoke(
    {"input": f"{PAPER_TEXT}"},
    config={"configurable": {"session_id": session_id}}
)

print("\n--- Final Result ---")
print(result['output'])

print("\n--- Final Chat History ---")
final_history = get_by_session_id(session_id)
for msg in final_history.messages:
    print(f"{type(msg).__name__}: {msg.content}")
    if isinstance(msg, AIMessage) and msg.tool_calls:
        print(f"  Tool Calls: {msg.tool_calls}")
    if isinstance(msg, ToolMessage):
        print(f"  Tool Call ID: {msg.tool_call_id}")

# store[session_id].clear()


with open("multi_agent_review_output_single_tool.txt", "w") as f:
    f.write("FINAL OUTPUT:\n")
    f.write(result["output"] + "\n\n")

    f.write("INTERMEDIATE STEPS:\n")
    for i, (agent_action, observation) in enumerate(result["intermediate_steps"], 1):
        f.write(f"Step {i}:\n")
        f.write(f"  Thought: {agent_action.log.strip()}\n")
        f.write(f"  Action Used: {agent_action.tool}\n")
        f.write(f"  Action Input: {agent_action.tool_input}\n")
        f.write(f"  Result: {observation.strip()}\n\n\n\n")


