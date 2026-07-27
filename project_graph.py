from langgraph.graph import StateGraph, START, END
from project_state import ProjectState
from planner_agent import _planner_agent


builder = StateGraph(ProjectState)

builder.add_node("planner_agent",_planner_agent)

builder.add_edge(START , "planner_agent")
builder.add_edge("planner_agent", END)


graph = builder.compile()

graph.invoke({"request": "Build a web application for online book reviews."})