from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool,wiki_tool, word_tool
import fastapi

load_dotenv() 

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
     [
         (
            "system",
            """
            Sei un assistente che aiuta gli operai di una fabbrica a capire il funzionamento di una macchina che si chiama dry cut.
            Li aiuterai a rispondere alle lore domande su allarmi setup macchina dati di produzione ecc...
            Per farlo usa principalmente il manuale che ti è stato fornito (Generale_nb7_v2_italiano_drycut).
            inserisci i dati in questo formato senza altro testo inutile.\n{format_instructions}
                
            """,
         ),
         ("placeholder","{chat_history}"),
         ("human","{query}"),
         ("placeholder","{agent_scratchpad}")
     ]
).partial(format_instructions = parser.get_format_instructions())

tools = [word_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(agent = agent, tools=tools, verbose=False)
query = input("Come posso aiutarti? ")
raw_respone = agent_executor.invoke({"query": query})


try:
    structured_response = parser.parse(raw_respone.get("output")[0]["text"])
    print(structured_response.summary)
except Exception as e:
    print("error parsing response", e, "Raw Response - ", raw_respone)

