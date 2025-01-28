from fastapi import FastAPI, Header
import uvicorn
import hashlib
from models.create_project import NewProject, Project
from models.project import TNTProject, ProjectAction
import signal
from threading import Timer
import os
from fastapi.middleware.cors import CORSMiddleware

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#ffd0260f2cdaa27329c5779cf4866cd1ed93c22e
root_project_key = hashlib.sha1(b"TNT").hexdigest()

from load_model.gpus import get_gpus, ChatRequest, ChatResponse

templates = Jinja2Templates(directory="panel")

@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest, x_project_key: str = Header(alias="X-Project-Key")):
    project = TNTProject(x_project_key)
    return get_gpus(request, project)

@app.post("/shutdown")
async def shutdown():
    print("Shutting down gracefully...")
    Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()

@app.post("/create_project")
async def create_project(request:  Project, x_project_key: str = Header(alias="X-Project-Key")):

    # Check if header X-Project-Key is correct
    if x_project_key != root_project_key:
        return {"error": "Invalid project key"}

    new_project = NewProject(request.name)

    if new_project.check_if_project_exists():
        return {"error": "Project already exists"}
    else:
        return {"message": new_project.create_project()}

@app.post("/project")
async def project(request: ProjectAction, x_project_key: str = Header(alias="X-Project-Key")):
    project = TNTProject(x_project_key)

    if not project.check_if_project_exists():
        return {"error": "Project does not exist"}
    else:
        method = request.method 

        if method == 'upload_file':
            return project.upload_file(request.file_name, request.file, request.file_format)
        elif method == 'edit_instructions':
            return project.edit_project_instructions(request.inst_text)
        elif method == 'get_instructions':
            return project.get_project_instructions()
        elif method == 'remove_file':
            return project.remove_file(request.file_name, request.file_format)
        elif method == 'get_files':
            return project.get_files()
        elif method == 'get_list_of_projects':
            return project.get_list_of_projects()
        elif method == 'get_list_of_conversations':
            return project.get_list_of_conversations()
        elif method == 'get_conversation':
            return project.get_conversation(request.conversation_id)
        elif method == 'delete_conversation':
            return project.delete_conversation(request.conversation_id)
        

        return {"error": "Invalid method"}

@app.get("/get_list_of_projects")
async def project_data(x_project_key: str = Header(alias="X-Project-Key")):
    project = TNTProject(123)
    return project.get_list_of_projects()

@app.get("/")
def read_root():
    server = {
        'name': 'tnt.server1',
        'version': '1.0.0',
    }

    return server

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)

