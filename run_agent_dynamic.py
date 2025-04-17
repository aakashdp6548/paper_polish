import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent, AgentType

# Set your OpenAI API key (or whichever LLM service you plan to use)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
llm = OpenAI(temperature=0.2)

# -------------------------
# Define the individual chains.
# -------------------------

# Strengths Chain: Given a paper text, outputs its strengths.
strengths_prompt = PromptTemplate(
    input_variables=["paper"],
    template="""
You are a seasoned machine learning conference reviewer.
Read the following paper and list its key strengths.
Consider aspects such as novelty, experimental rigor, clarity, and significance.

Paper:
{paper}

Strengths:"""
)

strengths_chain = LLMChain(llm=llm, prompt=strengths_prompt)

# Weaknesses Chain: Given a paper text, outputs its weaknesses.
weaknesses_prompt = PromptTemplate(
    input_variables=["paper"],
    template="""
You are a rigorous machine learning reviewer.
Read the following paper and list its weaknesses.
Focus on issues like experimental gaps, clarity problems, lack of related work discussion, or logical inconsistencies.

Paper:
{paper}

Weaknesses:"""
)

weaknesses_chain = LLMChain(llm=llm, prompt=weaknesses_prompt)

# Synthesis Chain: Given the paper text and its identified strengths and weaknesses, outputs improvement suggestions.
synthesis_prompt = PromptTemplate(
    input_variables=["paper", "strengths", "weaknesses"],
    template="""
You are a research mentor helping to improve a machine learning paper.
Given the paper below along with its identified strengths and weaknesses,
suggest concrete improvements that address the paper's weak points and build on its strengths.
Explain how the paper could be rewritten, how experiments could be extended, and how sections could be clarified.

Paper:
{paper}

Strengths:
{strengths}

Weaknesses:
{weaknesses}

Improvements:"""
)

synthesis_chain = LLMChain(llm=llm, prompt=synthesis_prompt)

# -------------------------
# Define the functions to be registered as tools.
# -------------------------

def strengths_tool(paper: str) -> str:
    """Tool to extract strengths from the paper text."""
    return strengths_chain.run(paper=paper)

def weaknesses_tool(paper: str) -> str:
    """Tool to extract weaknesses from the paper text."""
    return weaknesses_chain.run(paper=paper)

def synthesis_tool(paper: str) -> str:
    """
    Tool to synthesize improvements.
    
    This tool internally calls the other two functions to obtain the strengths and weaknesses,
    and then passes them to the synthesis chain.
    """
    # Get the intermediate results.
    strengths = strengths_tool(paper)
    weaknesses = weaknesses_tool(paper)
    # Now use the synthesis chain.
    return synthesis_chain.run(paper=paper, strengths=strengths, weaknesses=weaknesses)

# -------------------------
# Register these functions as tools for the agent.
# -------------------------

tools = [
    Tool(
        name="Strengths Analyzer",
        func=strengths_tool,
        description="Given a machine learning paper, returns its strengths."
    ),
    Tool(
        name="Weaknesses Analyzer",
        func=weaknesses_tool,
        description="Given a machine learning paper, returns its weaknesses."
    ),
    Tool(
        name="Synthesize Improvements",
        func=synthesis_tool,
        description="Given a machine learning paper, returns concrete improvements by leveraging the paper's strengths and weaknesses."
    )
]

# -------------------------
# Initialize a dynamic agent.
#
# Here we use LangChain’s Zero-Shot ReAct agent, which is capable of reasoning step-by-step
# and deciding which tool to call and in what order. The agent prompt will include
# instructions about the available tools.
# -------------------------

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# -------------------------
# Use the dynamic agent.
#
# The top-level input is the paper text.
# The agent will decide when to call the registered tools
# (for example, it might call "Synthesize Improvements" directly, or first query the "Strengths Analyzer"
# and "Weaknesses Analyzer" before synthesizing).
# -------------------------

def dynamic_review_and_improve(paper_text: str) -> str:
    """
    Use the dynamic agent to generate a final review/improvement.
    
    The agent internally queries its tools (the strengths and weaknesses analyzers and synthesis tool)
    to produce a comprehensive response.
    """
    final_response = agent.run(paper_text)
    return final_response

# -------------------------
# Example usage.
# -------------------------

if __name__ == '__main__':
    # For the demonstration, load your paper from a file (or any source)
    with open("paper.txt", "r", encoding="utf-8") as file:
        paper_text = file.read()
    
    # Get the final review and improvement suggestions using the dynamic agent.
    result = dynamic_review_and_improve(paper_text)
    print("Final Review and Improvement Suggestions:\n")
    print(result)