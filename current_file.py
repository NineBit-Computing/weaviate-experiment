import os
import weaviate
from dotenv import load_dotenv
from weaviate import Client as WeaviateClient
from weaviate.classes.init import Auth
from weaviate.auth import AuthApiKey
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Weaviate as WeaviateVectorStore
from langchain_community.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import cohere
import openai

load_dotenv()

weaviate_url = os.getenv("WEAVIATE_CLUSTER_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY") 
cohere_api_key = os.getenv("COHERE_API_KEY")
openai_api_key = os.getenv("OPEN_API_KEY")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,  
    auth_credentials=Auth.api_key(weaviate_api_key),  
    headers={'X-OpenAI-Api-key': openai_api_key} 
)

cohere_embeddings = CohereEmbeddings(api_key=cohere_api_key, model='command-r-plus-04-2024')

loader = PyPDFLoader("/home/khushi/weaviate/eco1.pdf")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
document_chunks = text_splitter.split_documents(docs)

vs = WeaviateVectorStore(
    client=client,
    index_name="LangChain_Weaviate_Index",
    embedding=cohere_embeddings,
    text_key="text",
)

for doc in document_chunks:
    vs.add_texts([doc])

template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences minimum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

question = "What is the moral of this story?"
retriever = vs.as_retriever()
context = retriever.get_relevant_documents(question)
context_text = " ".join([doc["text"] for doc in context])

formatted_prompt = prompt.format(question=question, context=context_text)

cohere_client = cohere.Client(api_key=cohere_api_key)

def cohere_generate(prompt_text):
    response = cohere_client.generate(
        model="command-xlarge",
        prompt=prompt_text,
        max_tokens=300,
        temperature=0.5,
    )
    return response.generations[0].text.strip()

# client.close()
response = cohere_generate(formatted_prompt)
print(response)
