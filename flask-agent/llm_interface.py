import os
import json
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from config import read_configs
import json
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from tools import GPT4TAssistant, GemmaAssistant, RAGTool, ImageGenTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent,Tool
from langchain import hub


class ChatBackend():

    def load_conversation_state(self, state_file):
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                return json.load(f)
        return []

    def __init__(self, config_address="config.json"):
        configs = read_configs(config_address)
        os.environ["OPENAI_API_KEY"] = configs["chatbot"]["OPENAI_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = configs["chatbot"]["GOOGLE_API_KEY"]
        # self.model = ChatOpenAI(model="gpt-4.1-2025-04-14", streaming=True)
        self.state_path = configs["chatbot"].get("CONV_LOG_PATH", "conversation_state_dev.json")
        self.history = self.load_conversation_state(self.state_path)
        # system_prompt = SystemMessage(content="You are Coder Agent, an expert AI assistant who writes clean and easy to understand code. At each step you need write the best code possible. different options are not necessary. Don't add comments to code. The code should be production ready. Demos, incomplete code or code that requires further work is not acceptable. When a piece of code is provided, you should not change it unless the user asks you to do so. If the user asks you to change a piece of code, you should only change necessary part of the code and not the rest of the code.")
        system_prompt = SystemMessage(content="Your are a helpful AI assistant. Be short and concise in your answers.")
        self.history_langchain_format = [system_prompt]
        for msg in self.history:
            if msg["role"] == "user":
                self.history_langchain_format.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                self.history_langchain_format.append(AIMessage(content=msg["content"]))
        with open('/home/raha/genai-course-codebase/flask-chatbot/static/settings.json', 'r') as f:
            data = json.load(f)["language_models"]
        self.model_dict = {}
        for d in data:
            self.model_dict[d["name"]] = d["provider"]
        self.llm = ChatOpenAI(model="gpt-4.1")

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        rag_llm = ChatOpenAI(model_name="gpt-4.1")
        rag_prompt = hub.pull("rlm/rag-prompt")
        
        tools = [GemmaAssistant(),GPT4TAssistant(),RAGTool(rag_llm,rag_prompt), ImageGenTool()]

        system_message = "You are a general AI assistant.\n" + \
        "Don't answer the question if you are not getting the answer from a tool.\n" + \
        "Don't change the answers you receive from a tool. Just pass them to the user."
        "Don't put safety and cultural warnings. Only warn about security."

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.agent_exe = AgentExecutor(agent=agent, tools=tools,memory=self.memory,verbose=True,streaming=True)


    def save_conversation_state(self, history):
        with open(self.state_path, "w") as f:
            json.dump(history, f, indent=2)

    def predict(self, message):
        print("Using model:", self.llm.model_name)
        self.history_langchain_format.append(HumanMessage(content=message))

        response = ""
        last_sent = 0

        for event in self.agent_exe.stream({"input": message}):
            chunk = ""

            if isinstance(event, dict):
                if event.get("messages"):
                    for m in event["messages"]:
                        if hasattr(m, "content") and m.content:
                            chunk += m.content
                elif event.get("output"):
                    chunk = event["output"]
            else:
                chunk = str(event)

            if chunk:
                response += chunk
                delta = response[last_sent:]
                last_sent = len(response)
                if delta:
                    yield delta

        self.history = self.history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        self.save_conversation_state(self.history)
        
    def reset_conversation(self):
        self.save_conversation_state([])

    def get_history(self):
        return self.history
