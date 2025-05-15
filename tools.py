import os
import logging
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict, Union
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from langchain_text_splitters import MarkdownTextSplitter
import magic
import docx2txt
from PIL import Image
import pytesseract
import textract
import uuid
from agents import function_tool
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gemini-embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/embedding-001")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 160))

# Validate environment variables
if not all([GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    raise ValueError("Missing required environment variables: GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY")

# Initialize clients
genai.configure(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

async def create_collection() -> None:
    """Create a Qdrant collection if it doesn’t exist and ensure payload index for 'file_path'."""
    logging.info("Attempting to create Qdrant collection")
    try:
        collections = qdrant_client.get_collections()
        if COLLECTION_NAME not in [c.name for c in collections.collections]:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            logging.info(f"Created collection '{COLLECTION_NAME}'.")
        else:
            logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
        
        # Ensure payload index for 'file_path'
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        if "file_path" not in collection_info.payload_schema:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="file_path",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logging.info("Created payload index for 'file_path'")
        else:
            logging.info("Payload index for 'file_path' already exists")
    except Exception as e:
        logging.error(f"Failed to create Qdrant collection or index: {str(e)}")
        raise

@function_tool
def qdrant_search(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    """Search the Qdrant vector store for relevant documents."""
    logging.info(f"qdrant_search called with query: {query}, top_k: {top_k}")
    try:
        resp = genai.embed_content(model=EMBED_MODEL, content=query, task_type="SEMANTIC_SIMILARITY")
        embedding = resp["embedding"]
        results = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=embedding, limit=top_k)
        retrieved = [{"text": p.payload.get("text", "")} for p in results.points]
        logging.info(f"Retrieved {len(retrieved)} results")
        return retrieved
    except Exception as e:
        logging.error(f"qdrant_search failed: {str(e)}")
        return [{"text": f"Error searching documents: {str(e)}"}]

def extract_text(file_path: str) -> str:
    """Extract text from various file types."""
    logging.info(f"Extracting text from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    mime = magic.from_file(file_path, mime=True)
    try:
        if mime == "application/pdf":
            reader = PyPDF2.PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif mime in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            return docx2txt.process(file_path)
        elif mime.startswith("image/"):
            return pytesseract.image_to_string(Image.open(file_path))
        else:
            raw = textract.process(file_path)
            return raw.decode("utf-8", errors="ignore")
    except Exception as e:
        logging.error(f"Text extraction failed: {str(e)}")
        raise ValueError(f"Unsupported file type or extraction error: {str(e)}")

@function_tool
async def create_embeddings(file_path: str, overwrite: bool = True) -> Dict[str, str]:
    """Create embeddings for a file and upsert to Qdrant, with option to overwrite duplicates."""
    logging.info(f"Processing file: {file_path}, overwrite: {overwrite}")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return {"status": "error", "message": "File not found"}
    try:
        if overwrite:
            filter_condition = Filter(
                must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
            )
            qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=filter_condition
            )
            logging.info(f"Deleted existing points for {file_path}")
        text = extract_text(file_path)
        if not text.strip():
            logging.warning("Extracted text is empty")
            return {"status": "error", "message": "No text extracted from the file"}
        chunks = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_text(text)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=genai.embed_content(model=EMBED_MODEL, content=chunk)["embedding"],
                payload={"text": chunk, "file_path": file_path}
            )
            for chunk in chunks
        ]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        logging.info(f"Added {len(points)} chunks from {file_path}")
        return {"status": "success", "message": f"Document added with {len(points)} chunks"}
    except Exception as e:
        logging.error(f"Embedding error: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

# tools.py
@function_tool
async def search_everything(query: str) -> Union[List[Dict[str, str]], Dict[str, str]]:
    """Perform a live web search using the Tavily API."""
    start_time = time.time()
    logging.info(f"Searching for: {query}")
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logging.error("No TAVILY_API_KEY found")
        return {"error": "No API key provided for search"}
    url = "https://api.tavily.com/search"
    payload = {"api_key": api_key, "query": query, "search_depth": "basic", "max_results": 5}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        results = [
            {"title": item.get("title", "").strip(), "snippet": item.get("content", "").strip()}
            for item in data.get("results", [])
        ]
        logging.info(f"Search returned {len(results)} results in {time.time() - start_time:.2f}s")
        return results
    except requests.exceptions.RequestException as e:
        logging.error(f"Search failed: {str(e)}")
        return {"error": f"Search failed: {str(e)}"}
