import os
import cohere
from dotenv import load_dotenv
import weaviate
from langchain_community.embeddings import CohereEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

load_dotenv()

# Environment Variables
weaviate_url = os.getenv("WEAVIATE_CLUSTER_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

# Cohere client initialization
cohere_client = cohere.Client(api_key=cohere_api_key)

# Initialize Weaviate client (using Weaviate v3 or v4)
client = weaviate.Client(
    url=weaviate_url,
    auth_client_secret=weaviate.auth.AuthApiKey(api_key=weaviate_api_key),
)

# Use Cohere embeddings for Weaviate
cohere_embeddings = CohereEmbeddings(api_key=cohere_api_key)

# Initialize Weaviate Vector Store with Cohere embeddings
vs = WeaviateVectorStore(
    client=client,
    index_name="LangChain_Weaviate_Index",  # Change the index name as needed
    embedding=cohere_embeddings,
    text_key="text",
)

# Example: Get documents from Weaviate
retriever = vs.as_retriever()

# Use retrieved documents for further tasks (e.g., QA)
question = "What is the moral of this story?"
context = retriever.get_relevant_documents(question)
context_text = " ".join([doc["text"] for doc in context])

# Use Cohere to generate the answer (with your custom prompt)
response = cohere_client.generate(
    model="command-xlarge",
    prompt=f"Question: {question}\nContext: {context_text}\nAnswer:",
    max_tokens=300,
    temperature=0.5,
)

# Print the generated response
print(response.generations[0].text.strip())
