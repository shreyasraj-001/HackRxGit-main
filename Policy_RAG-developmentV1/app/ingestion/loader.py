from docling.document_converter import DocumentConverter
import requests
import io
import pdfplumber as pp
import logging

import re
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
 
logging.basicConfig(level=logging.INFO)

def clean_address_text(text):
    """
    Clean address text using regex patterns
    """
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common address patterns that might be redundant
    text = re.sub(r'Premises No\.\s*\d+[-\d]*', '\n', text)
    text = re.sub(r'Plot no\.\s*[A-Z]+-\d+', '\n', text)
    
    # Clean up multiple commas and spaces
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing commas and spaces
    text = re.sub(r'^[,\s]+', '', text)
    text = re.sub(r'[,\s]+$', '', text)
    
    return text.strip()


def filter_insurance_text(obj):  
    if obj["object_type"] == "char":  
        unwanted_text = "National Insurance Co. Ltd. Premises No. 18-0374, Plot no. CBD-81, New Town, Kolkata - 700156"  
        if obj["text"].strip().lower() in unwanted_text.lower():  
            return False  
    return True  

class DocLoader:
    def __init__(self, file_link: str):
        self.file_link = file_link
        self.docling_document_converter = DocumentConverter()

    def docling_load(self):
        return self.docling_document_converter.convert(self.file_link)

    def pdf_plumber_load(self):
        pdf = requests.get(self.file_link)
        with pp.open(io.BytesIO(pdf.content)) as pdf:
            return pdf.pages



file_link = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"



# Extract the signature key
a = file_link.split("sig=")
sig_key = a[1]
print(f"Signature Key: {sig_key}")


# doc_loader = DocLoader(file_link).docling_load()
doc_loader = DocLoader(file_link).pdf_plumber_load()


data = ""
for page in doc_loader:
    doc = page.extract_text(layout=True)
    #filtered_doc =  page.filter(filter_insurance_text)
    # Clean the extracted text
    #cleaned_doc = clean_address_text(doc)
    data += doc

# data = doc_loader.document.export_to_text()

with open("agentic_doc_output.txt", "w") as f:
    f.write(data)

# print(doc_loader.load().pages)





# pc = PineconeVectorStore(index_name="hackrxn")

# pc.add_documents(data)

# from openai import OpenAI

# client = OpenAI(
#     base_url="https://api.cohere.ai/compatibility/v1",
#     api_key=os.getenv("COHERE_API_KEY"),
# )

# response = client.embeddings.create(
#     input=[data],
#     model="embed-v4.0",
#     encoding_format="float",
# )


# print(len(response.data[0].embedding))


from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
MODEL = "embed-v4.0"

embeddings = CohereEmbeddings(model=MODEL)

text_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = text_spliter.split_text(data)
#print(chunks)

print(len(chunks))
#print(chunks[0])

from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Convert strings to Document objects
documents = [Document(page_content=chunk) for chunk in chunks]

# Create Pinecone vector store with embeddings
pc = PineconeVectorStore.from_existing_index(
    #documents=documents,
    embedding=embeddings,
    index_name="policy-store",
    namespace=sig_key
)

print("Documents successfully added to Pinecone!")

# Create retriever without reranker
retriever = pc.similarity_search("What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?", k=20)

# Test the retriever
try:
    #results = retriever.invoke("What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?")
    print("Retrieval results:")
    for i, doc in enumerate(retriever):
        print(f"Document {i+1}: {doc.page_content[:200]}...")
except Exception as e:
    print(f"Error during retrieval: {e}")


# TODO: 
# 0. retrive the existing namespace if it exists pass one if not create one and add the documents to the namespace
# 1. Add a rerank
# 2. Add a semantic search
# 3. Add a vector store
# 4. Add a retriever
# 5. Add a query engine
# 6. Add a chain


