import os
from typing import TypedDict, Annotated, List, Literal, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# LLM initialization
from langchain.chat_models import init_chat_model
llm = init_chat_model("groq:llama-3.1-8b-instant")

# State Definition
class SupervisorState(MessagesState):
    next_agent: str = ""
    mission_plan: str = ""
    research_data: str = ""
    interpreted_insights: str = ""
    illustrations: str = ""
    task_complete: bool = False
    assignment: str = ""
    history_log: List[str] = []

# Supervisor Chain

def create_supervisor_chain():
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing a team of agents:\n\n1. Planner      - Breaks down the user task into sub-tasks\n2. Researcher   - Gathers information and data\n3. Interpreter  - Analyzes data and derives insights\n4. Illustrator  - Creates visuals and structured summaries\n\nBased on the current state and conversation, decide which agent should work next.\nIf the task is complete, respond with 'DONE'.\n\nCurrent state:\n- Has mission plan: {has_plan}\n- Has research data: {has_research}\n- Has interpreted insights: {has_insights}\n- Has illustrations/summary: {has_illustrations}\n\nRespond with ONLY the agent name (planner/researcher/interpreter/illustrator) or 'DONE'.\n"""),
        ("human", "{assignment}")
    ])
    return supervisor_prompt | llm

def supervisor_agent(state: SupervisorState) -> Dict:
    messages = state.get("messages", [])
    task = messages[-1].content if messages else "No task"
    has_plan = bool(state.get("mission_plan", ""))
    has_research = bool(state.get("research_data", ""))
    has_insights = bool(state.get("interpreted_insights", ""))
    has_illustrations = bool(state.get("illustrations", ""))
    chain = create_supervisor_chain()
    decision = chain.invoke({
        "assignment": task,
        "has_plan": has_plan,
        "has_research": has_research,
        "has_insights": has_insights,
        "has_illustrations": has_illustrations
    })
    decision_text = decision.content.strip().lower()
    if "done" in decision_text or has_illustrations:
        next_agent = "end"
        supervisor_msg = "✅ Supervisor: All tasks complete! Great work team."
    elif "planner" in decision_text or not has_plan:
        next_agent = "planner"
        supervisor_msg = "📋 Supervisor: Let's create a plan. Assigning to Planner..."
    elif "researcher" in decision_text or (has_plan and not has_research):
        next_agent = "researcher"
        supervisor_msg = "📋 Supervisor: Plan ready. Time for research. Assigning to Researcher..."
    elif "interpreter" in decision_text or (has_research and not has_insights):
        next_agent = "interpreter"
        supervisor_msg = "📋 Supervisor: Research done. Extracting insights. Assigning to Interpreter..."
    elif "illustrator" in decision_text or (has_insights and not has_illustrations):
        next_agent = "illustrator"
        supervisor_msg = "📋 Supervisor: Insights ready. Creating visuals/summary. Assigning to Illustrator..."
    else:
        next_agent = "end"
        supervisor_msg = "✅ Supervisor: Task seems complete."
    new_state = dict(state)
    new_state.update({
        "messages": [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": task
    })
    return new_state

def planner_agent(state: SupervisorState) -> Dict:
    task = state.get("assignment", "New Task")
    plan_prompt = f"""You are a planner. Break the following task into a clear step-by-step plan:\nTask: {task}\nProvide numbered steps and make it actionable for the next agents."""
    plan_response = llm.invoke([HumanMessage(content=plan_prompt)])
    mission_plan = plan_response.content
    agent_message = f"🗂️ Planner: I've created the mission plan for '{task}'.\n\nPlan:\n{mission_plan[:500]}..."
    new_state = dict(state)
    new_state.update({
        "messages": [AIMessage(content=agent_message)],
        "mission_plan": mission_plan,
        "next_agent": "supervisor",
        "history_log": state.get("history_log", []) + [agent_message],
    })
    return new_state

def researcher_agent(state: SupervisorState) -> Dict:
    task = state.get("assignment", "research topic")
    research_prompt = f"""As a research specialist, gather comprehensive information about: {task}\nInclude key facts, trends, statistics, and notable examples.\nUse concise and accurate language."""
    research_response = llm.invoke([HumanMessage(content=research_prompt)])
    research_data = research_response.content
    agent_message = f"🔍 Researcher: I've completed research on '{task}'.\n\nKey findings:\n{research_data[:500]}..."
    new_state = dict(state)
    new_state.update({
        "messages": [AIMessage(content=agent_message)],
        "research_data": research_data,
        "next_agent": "supervisor",
        "history_log": state.get("history_log", []) + [agent_message]
    })
    return new_state

def interpreter_agent(state: SupervisorState) -> Dict:
    research_data = state.get("research_data", "")
    interpret_prompt = f"Analyze and extract insights from:\n{research_data}"
    interpret_response = llm.invoke([HumanMessage(content=interpret_prompt)])
    interpreted_insights = interpret_response.content
    agent_message = f"🧠 Interpreter: Extracted insights.\n{interpreted_insights[:500]}..."
    new_state = dict(state)
    new_state.update({
        "messages": [AIMessage(content=agent_message)],
        "interpreted_insights": interpreted_insights,
        "next_agent": "supervisor",
        "history_log": state.get("history_log", []) + [agent_message],
    })
    return new_state

def illustrator_agent(state: SupervisorState) -> Dict:
    insights = state.get("interpreted_insights", "")
    illustrate_prompt = f"""You are an illustrator. Using the insights below, create a structured summary and suggest visuals or charts for key points:\n{insights}\nFormat clearly for presentation."""
    illustrate_response = llm.invoke([HumanMessage(content=illustrate_prompt)])
    illustrations = illustrate_response.content
    agent_message = f"🎨 Illustrator: Structured summary and suggested visuals ready.\n\n{illustrations[:500]}..."
    task_complete = all([
        state.get("mission_plan"),
        state.get("research_data"),
        state.get("interpreted_insights"),
        illustrations
    ])
    next_agent = "end" if task_complete else "supervisor"
    new_state = dict(state)
    new_state.update({
        "messages": [AIMessage(content=agent_message)],
        "illustrations": illustrations,
        "next_agent": next_agent,
        "task_complete": task_complete,
        "history_log": state.get("history_log", []) + [agent_message]
    })
    return new_state

from typing import Literal

def router(state: SupervisorState) -> Literal[
    "supervisor", "planner", "researcher", "interpreter", "illustrator", "__end__"
]:
    next_agent = state.get("next_agent", "supervisor")
    if next_agent == "end" or state.get("task_complete", False):
        return END
    if next_agent in ["supervisor", "planner", "researcher", "interpreter", "illustrator"]:
        return next_agent
    return "supervisor"

workflow = StateGraph(SupervisorState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("planner", planner_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("interpreter", interpreter_agent)
workflow.add_node("illustrator", illustrator_agent)
workflow.set_entry_point("supervisor")
for node in ["supervisor", "planner", "researcher", "interpreter", "illustrator"]:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "supervisor": "supervisor",
            "planner": "planner",
            "researcher": "researcher",
            "interpreter": "interpreter",
            "illustrator": "illustrator",
            "__end__": "__end__"
        }
    )
graph = workflow.compile()
