import re
import json
from pathlib import Path

from dotenv import load_dotenv
from typing import Any
from langchain_core.tools import tool

PROJECT_ROOT = Path("project_files")


def _build_tree(root: Path, structure: dict):
    for name, children in structure.items():
        # CHANGE: strip quote characters and anything else invalid in a
        # filename — mirrors the safe_name sanitization already used in
        # create_project_directory, applied here too since names can
        # arrive with stray quotes from JSON round-tripping
        safe_name = re.sub(r'[<>:"/\\|?*]', "", str(name)).strip()

        if not safe_name:
            continue  # skip anything that sanitizes down to nothing

        current = root / safe_name
        current.mkdir(parents=True, exist_ok=True)

        if isinstance(children, dict):
            _build_tree(current, children)


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

@tool
def create_file(workspace_path: str, path: str, content: str, overwrite: bool = True) -> dict:
    """
    Create a file with the given content, inside the given project workspace.

    Args:
        workspace_path: absolute path to the project root, as returned by
            create_project_directory. Every file MUST be written inside
            this workspace — never outside it.
        path: file path relative to workspace_path, e.g. 'backend/main.py'.
        content: full file content to write.
        overwrite: whether to overwrite an existing file at this path.

    Supports any text-based file such as:
    - .py, .java, .js, .ts, .tsx, .jsx, .html, .css, .scss, .json, .yaml,
      .yml, .xml, .sql, .md, .txt, Dockerfile, .gitignore,
      requirements.txt, package.json, pom.xml
    """
    root = Path(workspace_path)

    if not root.exists():
        return {
            "success": False,
            "message": f"workspace_path does not exist: {workspace_path}"
        }

    # guard against escaping the workspace via '..' or an absolute path
    if ".." in Path(path).parts or Path(path).is_absolute():
        return {
            "success": False,
            "message": f"rejected unsafe path: {path}"
        }

    file_path = root / path
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and not overwrite:
        return {
            "success": False,
            "message": f"{path} already exists."
        }

    file_path.write_text(content, encoding="utf-8")

    return {
        "success": True,
        "path": str(file_path.resolve())
    }

from typing import Any


@tool
def create_directory_tree(workspace_path: str, structure: Any) -> dict:
    """
    Create a nested directory tree inside the given project workspace.

    Args:
        workspace_path: absolute path to the project root, as returned by
            create_project_directory.
        structure: nested dict describing the folder layout, e.g.
            {"backend": {"models": {}, "routes": {}}, "frontend": {"src": {}}}
    """
    root = Path(workspace_path)

    if not root.exists():
        return {
            "success": False,
            "message": f"workspace_path does not exist: {workspace_path}"
        }

    # CHANGE: defend against structure arriving as a JSON-encoded string
    # rather than an actual dict — this is what produced the literal
    # quote characters in your error
    if isinstance(structure, str):
        try:
            structure = json.loads(structure)
        except json.JSONDecodeError:
            return {
                "success": False,
                "message": f"structure could not be parsed as JSON: {structure[:200]}"
            }

    if not isinstance(structure, dict):
        return {
            "success": False,
            "message": f"structure must be a dict, got {type(structure).__name__}"
        }

    _build_tree(root, structure)

    return {
        "success": True,
        "root": str(root.resolve())
    }