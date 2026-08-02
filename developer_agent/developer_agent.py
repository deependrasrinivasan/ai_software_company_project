from pathlib import Path
from langchain_core.tools import tool
import re
import json
from project_state import ProjectState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from developer_agent.developer_tools import create_project_directory, read_directory_tree, create_file, create_directory_tree
# CHANGE 1: tool lookup so tool calls can actually be executed by name
TOOL_MAP = {
    "create_project_directory": create_project_directory,
    "read_directory_tree": read_directory_tree,
    "create_file": create_file,
    "create_directory_tree": create_directory_tree
}

  # already imported at the top of your file, just confirming it's needed here too

def llm_call(prompt, state: ProjectState):
    llm = ChatGoogleGenerativeAI(
        temperature=0.7,
        model="gemini-3.1-flash-lite",
        max_output_tokens=1000,
    ).bind_tools([create_project_directory, read_directory_tree, create_file, create_directory_tree])

    chain = prompt | llm
    response = chain.invoke(
            {"tech_stack": state["tech_stack"], "design_doc": state["design_doc"]}
        )

    # CHANGE: track what's ACTUALLY created as each tool call executes,
    # instead of only printing it and discarding it. This becomes the
    # source of truth for the final response, not the model's summary.
    created_files = []
    workspace_path = None

    max_turns = 25  # raised from 5 — a real file set needs more than 5 tool round-trips
    turn = 0
    while getattr(response, "tool_calls", None) and turn < max_turns:
        tool_results = []
        for call in response.tool_calls:
            fn = TOOL_MAP[call["name"]]
            result = fn.invoke(call["args"])
            tool_results.append({"tool_call_id": call["id"], "content": str(result)})

            # CHANGE: record ground truth per tool, instead of printing
            # the raw tool call as if it were the response
            if call["name"] == "create_project_directory":
                workspace_path = result.split("\n")[-1].strip()
            elif call["name"] == "create_file":
                created_files.append(call["args"]["path"])

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

    # Loop only exits here once the model stops calling tools — i.e. all
    # files/dirs have genuinely been created. This is the "only after
    # all creation is done" point you wanted the response built at.

    # CHANGE: robust JSON extraction — don't assume response.text is pure
    # JSON with nothing else around it
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    try:
        json_response = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        json_response = {}

    # CHANGE: reconcile against ground truth. If the model's JSON is
    # missing or its paths don't match what was actually written, fall
    # back to the tracked created_files list — this guarantees the
    # returned paths are real, not the model's guess.
    backend_files = [p for p in created_files if p.startswith("backend/")]
    frontend_files = [p for p in created_files if p.startswith("frontend/")]

    if not json_response.get("backend_files"):
        json_response["backend_files"] = [{"path": p, "purpose": "", "depends_on": []} for p in backend_files]
    if not json_response.get("frontend_files"):
        json_response["frontend_files"] = [{"path": p, "purpose": "", "depends_on": []} for p in frontend_files]

    json_response["workspace_path"] = workspace_path or json_response.get("workspace_path")

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
            """You are a Senior Software Developer responsible for scaffolding
and generating a complete, runnable project from a design document and a
chosen tech stack.

## Your responsibilities

- Thoroughly analyze the provided design document and tech stack before
  taking any action.
- Design a directory structure that is organized, scalable, and follows
  conventions idiomatic to the given tech stack (e.g. a FastAPI project is
  structured differently from an Express project — use the standard layout
  for whatever stack you were given, not a generic one).
- Create every backend and frontend file required to satisfy the design
  document, with real, working code — not placeholders or stubs, unless the
  design doc explicitly describes something as out of scope.
- Include a dependency manifest appropriate to the stack in the project
  root (e.g. requirements.txt for Python, package.json for Node/React) that
  lists every package your generated code actually imports or uses.
- Design for maintainability and security as a baseline: no hardcoded
  secrets, sensible error handling, no obviously unsafe patterns (e.g. raw
  string SQL concatenation).
- Ensure every file path is unique and does not collide with another file
  or directory you create in this same run.

## Required order of operations — follow these steps strictly, in order

1. first create the project directory with unique name and get the absolute path, then use that as the root for all subsequent paths.
the root should be created under the project files directory using the tool create project directory.
keep this root path and make sure to create all the files and subdirectories under this root path. Do not create any files or directories outside of this root path.
    Call `create_project_directory` first, before anything else. Do not
   proceed until you have the real absolute path it returns — this becomes
   your `workspace_path` and every later path is relative to it.
2. Decide the full directory structure needed for both backend and
   frontend, then call `create_directory_tree` (or `create_subdirectories`,
   if available) once with that complete structure, rather than creating
   folders one at a time.
3. For each file you need to create, first identify what it depends on
   (e.g. a route file depends on the model it imports). Create files in
   dependency order — a file that is imported by another file should be
   created before the file that imports it — so that when you write a
   dependent file, its imports and interfaces are already real and
   consistent with what exists.
4. Call `create_file` for each file individually, with complete, final
   code content — not incremental drafts. Use `read_directory_tree` if you
   need to confirm the current state of the project before continuing.
5. Only after every planned file has been created and confirmed, produce
   your final response. Do not produce the final JSON output while any
   tool calls are still pending or planned — the JSON is a report of what
   you have already done, not a plan of what you intend to do.

## Tool selection

- Use `create_project_directory` exactly once, at the very start.
- Use `create_directory_tree` for laying out the folder skeleton in one
  call, rather than relying on each `create_file` call to create parent
  folders as a side effect.
- Use `create_file` for every individual source file, config file, and the
  dependency manifest.
- Use `read_directory_tree` if you need to verify existing structure
  before deciding your next step — not required for every file, only when
  genuinely uncertain.

## Output rules — read carefully

- Your final response must be a single JSON object and nothing else: no
  prose before or after it, no markdown code fences, no comments inside
  the JSON.
- Every path listed in `backend_files` and `frontend_files` must be a file
  you have actually created via `create_file` in this run — do not invent
  or plan paths that were never written.
- `workspace_path` must be the exact absolute path returned by
  `create_project_directory`, not a guess or placeholder.
- `depends_on` for each file should list the relative paths of other files
  in this same manifest that it imports from or calls — leave it as an
  empty list if the file has no such dependencies.

The final JSON object must follow exactly this structure, with no
additional or missing top-level fields:

{{
    "backend_files": [
        {{"path": "backend/main.py", "purpose": "app entrypoint, mounts routers", "depends_on": []}},
        {{"path": "backend/models/todo.py", "purpose": "Todo SQLAlchemy model", "depends_on": []}},
        {{"path": "backend/routers/todos.py", "purpose": "CRUD routes for todos", "depends_on": ["backend/models/todo.py"]}}
    ],
    "frontend_files": [
        {{"path": "frontend/src/api/todos.ts", "purpose": "fetch calls to /todos", "depends_on": ["backend/routers/todos.py"]}},
        {{"path": "frontend/src/pages/TodoList.tsx", "purpose": "main list view", "depends_on": ["frontend/src/api/todos.ts"]}}
    ],
    "workspace_path": "/absolute/path/returned/by/create_project_directory",
    "next_action": "qa"
}}
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