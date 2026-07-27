from typing import List, TypedDict

class ProjectState(TypedDict, total=False):
    request: str
    requirements: str
 
    design_doc: str
    tech_stack: str
 
    backend_code: str
    frontend_code: str
 
    test_results: str
    bug_reports: List[str]
 
    deployment_url: str
    docs: str
 
    errors: List[str]
    next_action: str        # routes conditional edges
    retry_count: int        # guards against infinite QA <-> Development loops
    current_department: str # set at entry of each node, used by error_handler