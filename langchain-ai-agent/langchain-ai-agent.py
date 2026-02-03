from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_tools import ddg_tool, wikipedia_tool, txt_file_tool
from datetime import datetime


# Load environment variables
load_dotenv()

# Instantiate Gemini instance
gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# Specifying the fields expected from the LLM output
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Creating an agent
agent = create_agent(
    model = gemini,
    response_format=ResearchResponse,
    tools=[ddg_tool, wikipedia_tool, txt_file_tool]
)

# Gathering query from the user 
user_query = input("What topic do you want researched? ") 

# Unique filename
timestamp = datetime.now().strftime("%Y-%m-%d")
filename = f"ai-research-output-{timestamp}"

# Running the AI agent
raw_response = agent.invoke(
    {
        "messages": [
            {
                "role": "user", 
                "content": f"""
                            Research this topic: {user_query}. 
                            After completing your research, you MUST save the results by calling txt_file_tool with:
                            - llm_output: your complete research findings
                            - filename: '{filename}'
                            """
            }
        ]
    }
)

# print(raw_response, end='\n\n')
print(raw_response['structured_response'], end='\n\n')
print(raw_response['structured_response'].tools_used, end='\n\n')