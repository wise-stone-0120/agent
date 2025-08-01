# 1. requirements.txt 는 파이선 모듈들을 적어놓는다.
# 2. 파일명도 대소문자 구분한다. 못 찾네.
# 3. .env는 강제로 지정해야 한다. uv run dotenv -f ../.env run -- uv run gradio deploy
# 4. huggingface space website에서 api 들을 secret으로 등록하고 성공했다.
# 5. 1_foundation 폴더 전체를 hugginface에서 space에 올려 놓고 가성 환경을 만들어준다. \
#    따라서, 필요한 파일 및 자원은 해당 폴더 안에 있어야 한다.

# import
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import json
import requests
import gradio as gr

load_dotenv(override = True)

# push over 함수
def push(text):
    payload = {
            "user":os.getenv("PUSHOVER_USER"),
            "token": os.getenv("PUSHOVER_TOKEN"),
            "message": text
        }
    requests.post("https://api.pushover.net/1/messages.json",
                  data=payload)

# tool
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recored":"ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded":"ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recoding to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            }
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type":"function", "function": record_user_details_json},
    {"type":"function", "function": record_unknown_question_json}
]

# class 정의
class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Inseok Park"
        reader = PdfReader("me/InseokPark.pdf")
        self.linkedin = ""
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                self.linkedin += txt
        with open("me/aboutMe.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()
        
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
            particularly questions related to {self.name}'s career, background, experience. \
            Your responsibility is to represent {self.name} for interaction on the website as faithfully as possilbe. \
            You're given {self.name}'s background summary and LinkedIn profile which you can use to answer questions. \
            Be professional and engaging, as if potential client and future employer who came across the website. \
            If you don't know the answer to any question, user your record_unknown_question tool to record the question \
            that you couldn't answer, even if it is about something trivial or unrelated to career. \
            If the user is engaging in discussion, try to steer them toward being in touch via email; ask for their email \
            and record it using your record_user_details tool. "
        
        system_prompt += f"\n\nSummary: {self.summary}\n\nLinkedIn Profile: {self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always stay in character as {self.name}."
        return system_prompt

    def handle_tool_call(self, tool_calls):
        results=[]
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role":"tool", "content":json.dumps(result), "tool_call_id":tool_call.id})

        return results
    
    def chat(self, message, history):
        messages = [{"role":"system", "content":self.system_prompt()}] + history + [{"role":"user", "content":message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
				tools=tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done=True
        return response.choices[0].message.content

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()