import os
import json
from pydantic import BaseModel
from datetime import datetime
import base64

class ProjectAction(BaseModel):
    method: str
    file: str | None = None
    file_name: str | None = None
    file_format: str | None = None
    instructions: str | None = None
    inst_text: str | None = None
    
    class Config:
        arbitrary_types_allowed = True


class TNTProject:
    def __init__(self, project_key: str):
        self.project_key = project_key
        self.project = f"projects/{project_key}"

    def check_if_project_exists(self):
        if os.path.exists(self.project):
            return True
        else:
            return False

    def get_project_json(self):
        with open(self.project+'/project.json', 'r') as f:
            return json.load(f)
    
    def get_project_instructions(self):
        with open(self.project+'/instruction.txt', 'r') as f:
            return f.read()
    
    def edit_project_instructions(self, instructions: str):
        with open(self.project+'/instruction.txt', 'w') as f:
            f.write(instructions)

        project_data = self.get_project_json()
        project_data['instructions_last_modified'] = datetime.now().isoformat()
        with open(self.project+'/project.json', 'w') as f:
            json.dump(project_data, f)
        return {"message": "Instructions edited successfully"}

    def upload_file(self, file_name: str, file: str, file_format: str):
        file_path = f"{file_name}.{file_format}"
        content_decode_base64 = base64.b64decode(file)
        if self.check_if_file_data_exists(file_path):
            return {"error": "File already exists"}
        else:
            with open(self.project+'/data/'+file_path, 'wb') as f:
                f.write(content_decode_base64)
            self.write_in_project_json_file(file_path)
            return {"message": "File uploaded successfully"}
        
    def remove_file(self, file_name: str, file_format: str):
        file_path = f"{file_name}.{file_format}"    
        if not self.check_if_file_data_exists(file_path):
            return {"error": "File does not exist"}
        else:
            os.remove(self.project+'/data/'+file_path)
            self.remove_file_from_project_json_file(file_path)
            return {"message": "File removed successfully"}

    def get_files(self):
        return self.get_project_json()['files']

    def check_if_file_data_exists(self, file_name: str):
        if os.path.exists(self.project+'/data/'+file_name):
            return True
        else:
            return False
        
    def write_in_project_json_file(self, filename: str):
        project_data = self.get_project_json()
        data = {'file': filename, 'created_at': datetime.now().isoformat()}
        project_data['files'].append(data)
        with open(self.project+'/project.json', 'w') as f:
            json.dump(project_data, f)

    def remove_file_from_project_json_file(self, filename: str):
        project_data = self.get_project_json()
        project_data['files'] = [file for file in project_data['files'] if file['file'] != filename]
        with open(self.project+'/project.json', 'w') as f:
            json.dump(project_data, f)

    def get_list_of_projects(self):
        projects1 = os.listdir('projects')
        #open json get name, created at, and key
        projects = []
        for project in projects1:
            with open(f'projects/{project}/project.json', 'r') as f:
                project_data = json.load(f)
                project_data['name'] = project_data['name']
                project_data['created_at'] = project_data['created_at']
                project_data['key'] = project_data['key']
                projects.append(project_data)
        return projects
    
    def check_if_conversation_exists(self, conversation_id: str):
        if os.path.exists(self.project+'/conversation/'+conversation_id):
            return True
        else:
            return False
    
    def create_conversation(self, conversation_id: str):
        if self.check_if_conversation_exists(conversation_id):
            return conversation_id
        else:
            conversation_id = self.create_conversation_id()
            os.makedirs(self.project+'/conversation/'+conversation_id)
            return conversation_id
        
    def create_conversation_id(self):
        return str(hash(str(datetime.now().isoformat())))[:10]
    
    def save_conversation(self, conversation_id: str, messages: list):
        with open(self.project+'/conversation/'+conversation_id, 'w') as f:
            json.dump(messages, f)

    def get_conversation(self, conversation_id: str):
        with open(self.project+'/conversation/'+conversation_id, 'r') as f:
            return json.load(f)
