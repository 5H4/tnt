import os
import json
from pydantic import BaseModel
import hashlib
from datetime import datetime

class Project(BaseModel):
    name: str


class NewProject:

    def __init__(self, name: str):
        self.name = name
        self.project = 'projects/'

    def check_if_project_exists(self):

        project_key = self.create_project_key()

        if os.path.exists(self.project+project_key):
            return True
        else:
            return False
        
    def create_project(self):

        project_key = self.create_project_key()

        os.makedirs(self.project+project_key)
        os.makedirs(self.project+project_key+'/data')

        project_data = {
            'name': self.name,
            'key': project_key,
            'created_at': datetime.now().isoformat(),
            'instructions_path': 'instruction.txt',
            'instructions_last_modified': datetime.now().isoformat(),
            'files': []
        }

        with open(self.project+project_key+'/project.json', 'w') as f:
            json.dump(project_data, f)

        with open(self.project+project_key+'/instruction.txt', 'w') as f:
            f.write("This is a project instructions file. You can edit it to add your own instructions.")

        return project_key

    def create_project_key(self):
        return hashlib.sha1(self.name.encode()).hexdigest()

