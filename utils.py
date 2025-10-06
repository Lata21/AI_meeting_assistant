import requests
import re
import json
from datetime import datetime, timedelta


class TodoistTools:
    def __init__(self, api_token):
        self.api_token = api_token
        self.base_url = "https://app.todoist.com/rest/v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def get_projects(self):
        response = requests.get(f"{self.base_url}/projects", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed to get projects: {response.status_code}"}

    def get_project(self, project_name):
        projects = self.get_projects()
        if "error" in projects:
            return projects
        for project in projects:
            if project["name"].lower() == project_name.lower():
                return project
        return None

    def create_project(self, project_name, color="berry_red"):
        existing_project = self.get_project(project_name)
        if existing_project:
            return existing_project
        data = {"name": project_name, "color": color}
        response = requests.post(f"{self.base_url}/projects", headers=self.headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed to create project: {response.status_code}"}

    def get_collaborators(self, project_id):
        response = requests.get(f"{self.base_url}/projects/{project_id}/collaborators", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed to get collaborators: {response.status_code}"}

    def create_task(self, content, project_id, due_string=None, priority=3, assigned_id=None):
        data = {
            "content": content,
            "project_id": project_id,
            "priority": priority
        }
        if due_string:
            data["due_string"] = due_string
        if assigned_id:
            data["assigned_id"] = assigned_id
        response = requests.post(f"{self.base_url}/tasks", headers=self.headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed to create task: {response.status_code}"}

    def create_and_assign_task(self, content, project_name, assignee_name=None, due_string=None, priority=3):
        project = self.get_project(project_name)
        if not project:
            project = self.create_project(project_name)
            if "error" in project:
                return project
        project_id = project["id"]

        assignee_id = None
        if assignee_name:
            collaborators = self.get_collaborators(project_id)
            if "error" not in collaborators:
                for collaborator in collaborators:
                    if collaborator["name"].lower() == assignee_name.lower():
                        assignee_id = collaborator["id"]
                        break

        return self.create_task(content, project_id, due_string, priority, assignee_id)


class TranscriptExtractor:
    def __init__(self, source_type="google_meet"):
        self.source_type = source_type

    def get_transcript(self, meeting_id):
        if self.source_type == "google_meet":
            return self._get_google_meet_transcript(meeting_id)
        elif self.source_type == "whatsapp":
            return self._get_whatsapp_transcript(meeting_id)
        elif self.source_type == "telegram":
            return self._get_telegram_transcript(meeting_id)
        return {"error": f"Unsupported source type: {self.source_type}"}

    def _get_google_meet_transcript(self, meeting_id):
        return {
            "meeting_id": meeting_id,
            "transcript": "This is a sample transcript from Google Meet.",
            "participants": ["Avni Bharti", "Deepali Pateriya", "Lakhan Jadhwant"]
        }

    def _get_whatsapp_transcript(self, chat_id):
        return {
            "chat_id": chat_id,
            "transcript": "This is a sample transcript from WhatsApp.",
            "participants": ["Avni Bharti", "Deepali Pateriya", "Lakhan Jadhwant"]
        }

    def _get_telegram_transcript(self, chat_id):
        return {
            "chat_id": chat_id,
            "transcript": "This is a sample transcript from Telegram.",
            "participants": ["Avni Bharti", "Deepali Pateriya", "Lakhan Jadhwant"]
        }


class TelegramCommunicator:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(self, message):
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(f"{self.base_url}/sendMessage", data=data)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed to send message: {response.status_code}"}

    def ask_confirmation(self, question, options=None):
        if options is None:
            options = ["Yes", "No"]
        keyboard = [[{"text": option, "callback_data": option}] for option in options]

        data = {
            "chat_id": self.chat_id,
            "text": question,
            "reply_markup": json.dumps({"inline_keyboard": keyboard})
        }

        response = requests.post(f"{self.base_url}/sendMessage", data=data)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed to send message: {response.status_code}"}


class TaskExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract_tasks_from_transcript(self, transcript):
        """Extract tasks from transcript using llm"""
        prompt = f"""
Please analyze the following meeting transcript and identify:
1. Project names mentioned
2. Tasks that need to be completed
3. Who should be assigned to each task (if mentioned)
4. Due dates for tasks (if mentioned)

Format your response as JSON with the following structure:

{{
  "projects": [
    {{
      "name": "Project Name",
      "tasks": [
        {{
          "content": "Task description",
          "assignee": "Assignee Name or null",
          "due_string": "Due date string or null",
          "priority": 1
        }}
      ]
    }}
  ]
}}

Transcript:
{transcript}
"""

        # Invoke LLM
        response = self.llm.invoke(prompt).content

        # Extract JSON block (if enclosed in ```json ... ```)
        json_match = re.search(r'```json\n*(.*?)\n*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response

        try:
            clean_json = json.str.strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            return {"error": "Failed to parse extracted tasks"}
        


class TodoistMeetingManager:
    def __init__(self, todoist_api_token, telegram_bot_token=None, telegram_chat_id=None,
                 transcript_souce="google_meet", llm=None):
        self.todoist_tools = TodoistTools(todoist_api_token)
        self.transcript_extractor = TranscriptExtractor(transcript_souce)
        self.telegram = None
        if telegram_bot_token and telegram_chat_id:
            self.telegram = TelegramCommunicator(telegram_bot_token, telegram_chat_id)
        self.task_extractor = TaskExtractor(llm)

    def process_meeting(self, meeting_id):
        """Process a meeting and create tasks in Todoist"""
        transcript_data = self.transcript_extractor.get_transcript(meeting_id)
        if "error" in transcript_data:
            return transcript_data

        extracted_data = self.task_extractor.extract_tasks_from_transcript(transcript_data["transcript"])
        if "error" in extracted_data:
            return extracted_data

        results = {
            "projects_created": [],
            "tasks_created": []
        }

        for project_data in extracted_data["projects"]:
            project_name = project_data["name"]

            create_project = True
            if self.telegram:
                confirmation = self.telegram.ask_confirmation(
                    f"Should I create a new project called *{project_name}*?"
                )
                # You may implement actual confirmation logic here.

            # Create project in Todoist
            project = self.todoist_tools.create_project(project_name)
            if "error" in project:
                continue  # skip if creation failed

            results["projects_created"].append(project_name)

            for task_data in project_data["tasks"]:
                content = task_data["content"]
                assignee = task_data.get("assignee")
                due_string = task_data.get("due_string")
                priority = task_data.get("priority", 3)

                task_result = self.todoist_tools.create_and_assign_task(
                    content,
                    project_name,
                    assignee,
                    due_string,
                    priority
                )

                if "error" not in task_result:
                    results["tasks_created"].append(task_result["content"])
                else:
                    results["tasks_created"].append({
                        "task": content,
                        "error": task_result["error"]
                    })

        # Send summary via Telegram
        if self.telegram:
            message = "*Meeting Tasks Processed Successfully*\n\n"
            message += "*Projects Created:*\n" + "\n".join(f"- {p}" for p in results["projects_created"]) + "\n\n"
            message += "*Tasks Created:*\n" + "\n".join(
                f"- {t}" if isinstance(t, str) else f"- {t['task']} (Error: {t['error']})"
                for t in results["tasks_created"]
            )
            self.telegram.send_message(message)

        return results
