from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index name
index_name = "medicalbot"

# Check if index already exists
existing_indexes = [index["name"] for index in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Creating a new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # Ensure the embedding model matches this dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index '{index_name}' already exists. Using existing index.")

# Load PDF data and split into chunks
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)

# Download Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Use existing index if available
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Add documents to the Pinecone index
docsearch.add_documents(documents=text_chunks)

print(f"Indexing completed for {len(text_chunks)} text chunks!")
