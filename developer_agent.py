from pathlib import Path
from langchain_core.tools import tool
import re
import json
from project_state import ProjectState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

PROJECT_ROOT = Path("project_files")


@tool
def create_project_directory(project_name: str) -> str:
    """
    Creates a new project directory inside 'project_files'.

    Args:
        project_name: Name of the project.

    Returns:
        Absolute path of the created project directory.
    """

    # Create root folder if it doesn't exist
    PROJECT_ROOT.mkdir(exist_ok=True)

    # Make the project name filesystem-safe
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', "_", project_name.strip())

    project_path = PROJECT_ROOT / safe_name

    if project_path.exists():
        return f"Project already exists at:\n{project_path.resolve()}"

    project_path.mkdir(parents=True)

    return f"Project created successfully:\n{project_path.resolve()}"

@tool
def read_directory_tree(path: str) -> dict:
    """
    Return the complete directory tree.
    """

    root = Path(path)

    if not root.exists():
        return {
            "success": False,
            "message": "Directory not found."
        }

    tree = []

    for item in sorted(root.rglob("*")):
        tree.append({
            "path": str(item.relative_to(root)),
            "type": "directory" if item.is_dir() else "file"
        })

    return {
        "success": True,
        "root": str(root.resolve()),
        "tree": tree
    }

# CHANGE 1: tool lookup so tool calls can actually be executed by name
TOOL_MAP = {
    "create_project_directory": create_project_directory,
    "read_directory_tree": read_directory_tree,
}


def llm_call(prompt , state: ProjectState):
    # CHANGE 2: tools passed via .bind_tools(), not the constructor's tools= kwarg
    llm = ChatGoogleGenerativeAI(
        temperature=0.7,
        model="gemini-3.1-flash-lite",
        max_output_tokens=1000,
    ).bind_tools([create_project_directory, read_directory_tree])

    chain = prompt | llm
    response = chain.invoke(
            {"tech_stack": state["tech_stack"], "design_doc": state["design_doc"]}
        )

    messages = chain.first.invoke_messages if False else None  # (kept out, see note below)

    max_turns = 5
    turn = 0
    while getattr(response, "tool_calls", None) and turn < max_turns:
        tool_results = []
        for call in response.tool_calls:
            fn = TOOL_MAP[call["name"]]
            result = fn.invoke(call["args"])
            tool_results.append({"tool_call_id": call["id"], "content": str(result)})
            print(f"\n[tool call] {call['name']}({call['args']}) -> {result}\n")

        # re-invoke the model with the tool results appended so it can
        # continue (e.g. now produce the final JSON using the real path)
        response = llm.invoke(
            [
                {"role": "user", "content": prompt.format(
                    tech_stack=state["tech_stack"], design_doc=state["design_doc"]
                )},
                response,
                *[{"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]} for r in tool_results],
            ]
        )
        turn += 1

    json_response = json.loads(response.text)
    print("\n========== front end ==========\n")
    print(json_response["frontend_files"])
    
    print("\n========== backend ==========\n")
    print(json_response["backend_files"])

    return json_response

def _developer_agent(state:ProjectState):
    developer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """ Your role is to act as a Senior Software Developer.
                from the design document and tech stack provided, you will write the backend and frontend code for the project.

                Your responsibilities:
                - Analyze the provided design document and tech stack thoroughly.
                -you wil be only creating the directories for the project and not the files.
                - Create a complete project directory structure based on the design document and tech stack.
                - Ensure the directory structure is organized, scalable, and follows best practices.
                -the directory names should be unique and should not conflict with existing directories.

                Output Rules:
                - You MUST call the create_project_directory tool first to actually
                create the project directory before producing any output.
                -based on the tools provided, you will first create the project directory structure and return the absolute path of the created project directory in the workspace_path field of the json object just like mentioned below.

                the final output should be a json object with the following structure:
                {{
    "backend_files": [
        {{"path": "backend/main.py", "purpose": "app entrypoint, mounts routers", "depends_on": []}},
        {{"path": "backend/models/todo.py", "purpose": "Todo SQLAlchemy model", "depends_on": []}},
        {{"path": "backend/routers/todos.py", "purpose": "CRUD routes for todos", "depends_on": ["backend/models/todo.py"]}},
    ],
    "frontend_files": [
        {{"path": "frontend/src/api/todos.ts", "purpose": "fetch calls to /todos", "depends_on": ["backend/routers/todos.py"]}},
        {{"path": "frontend/src/pages/TodoList.tsx", "purpose": "main list view", "depends_on": ["frontend/src/api/todos.ts"]}},
    ],
    "workspace_path": "/tmp/projects/demo-1",
    "next_action": "qa",
    # unchanged from input, still present: request, requirements, design_doc, tech_stack, thread_id
                }}
        strictly maintain the json structure and do not add any additional fields or comments.
                
                
                """

            ),
            (
                "user",
                "tech_stack: {tech_stack}\ndesign_doc: {design_doc}"
            )
        ]
    )
    response = llm_call(developer_prompt, state)


    return {
        "backend_code": response["backend_files"],
        "frontend_code": response["frontend_files"]
    }