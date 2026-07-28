from langgraph.graph import StateGraph, START, END
from architect_agent import _architect_agent
from project_state import ProjectState
from planner_agent import _planner_agent


builder = StateGraph(ProjectState)

builder.add_node("planner_agent",_planner_agent)
builder.add_node("architect_agent",_architect_agent)
builder.add_edge(START , "planner_agent")
builder.add_edge("planner_agent", "architect_agent")
builder.add_edge("architect_agent", END)


graph = builder.compile()

graph.invoke({"request": "Build a web application for online book reviews."})