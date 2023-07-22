import os
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT
from constants import PINECONE_INDEX_NAME
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

# Set API key for OpenAI and Pinecone
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENVIRONMENT
)

# Load PDF
loader = UnstructuredPDFLoader(
    'How_Conversational_Business_Can_Help_You_Get_and_Stay_Closer_to_Customers.pdf')

data = loader.load()

print(f'You have loaded a PDF with {len(data)} pages')
print(f'There are {len(data[0].page_content)} characters in your document')

# Chunk data into smaller documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(data)

print(f'You have split your document into {len(texts)} smaller documents')

# Create embeddings from your documents
embeddings = OpenAIEmbeddings(client='')
docsearch = Pinecone.from_texts(
    [t.page_content for t in texts], embeddings, index_name=PINECONE_INDEX_NAME)
