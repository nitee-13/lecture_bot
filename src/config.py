"""
Configuration settings for the LectureBOT application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# App settings
APP_TITLE = "LectureBOT"
APP_DESCRIPTION = "A personalized chatbot for lecture slides"

# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Vector database settings
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/embeddings")

# Knowledge graph settings
KNOWLEDGE_GRAPH_PATH = os.getenv("KNOWLEDGE_GRAPH_PATH", "data/knowledge_graph")

# PDF storage path
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "data/pdfs")

# Embedding dimensions
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 768))

# Chunk size for text splitting
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Model settings
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# UI Settings
NODE_SIZE = 25
EDGE_LENGTH = 100
TOP_K_RESULTS = 5

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(KNOWLEDGE_GRAPH_PATH, exist_ok=True)

# Embedding settings
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Model settings
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# UI Settings
APP_TITLE = "Lecture Bot"
APP_EMOJI = "📚"
APP_DESCRIPTION = "Your personal lecture assistant powered by RAG and Gemini 2.0"

# Retrieval settings
TOP_K_RESULTS = 5  # Number of chunks to retrieve for RAG

# Prompt settings
SYSTEM_PROMPT = """You are a helpful teaching assistant that helps students understand lecture materials. 
You have access to lecture slides and can answer questions about the content.
Your responses should be accurate, clear, and based on the provided context.
When mathematical equations are involved, format them properly.
When you're not sure about the answer, admit it rather than making up an answer."""

# Knowledge graph settings
KG_NODE_SIZE = 25
KG_EDGE_LENGTH = 100
KG_WIDTH = "100%"
KG_HEIGHT = "600px"
