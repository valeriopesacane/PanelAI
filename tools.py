from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

loader = Docx2txtLoader("Generale_NB7_V2_ITALIANO_DRYCUT 4.0.docx")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

def query_word_file(query: str) -> str:
    """Cerca nel documento word una risposta alla query."""
    results = retriever.invoke(query)
    if not results:
        return "No relevant info found in Word file."
    return "\n".join([doc.page_content[:300] for doc in results])

word_tool = Tool(
    name="ManualeWord",
    func=query_word_file,
    description="Utile per rispondere alle domandi riguardanti la macchina leggendo il file word Generale_NB7_V2_ITALIANO_DRYCUT 4.0.docx "
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_web",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)