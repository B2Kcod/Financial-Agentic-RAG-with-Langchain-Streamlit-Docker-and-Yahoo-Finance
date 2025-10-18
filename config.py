from dotenv import load_dotenv
from pathlib import Path
import os
import ast

# Load environment variables from .env file.
load_dotenv()

# for loader
DATA_PATH             = Path(os.getenv("DATA_PATH"))
TOC_PATH              = Path(os.getenv("TOC_PATH"))
DATA_TYPE             = os.getenv("DATA_TYPE")

# for splitter
CHUNK_SIZE            = int(os.getenv("CHUNK_SIZE"))   
CHUNK_OVERLAP         = int(os.getenv("CHUNK_OVERLAP"))  
SEPARATORS            = ast.literal_eval(os.getenv("SEPARATORS"))

# for encoder
ENCODER_NAME          = os.getenv("ENCODER_NAME")   

# for embedder
EMBEDDER_NAME         = os.getenv("EMBEDDER_NAME")

# for vectorstores
PERSIST_DIRECTORY     = Path(os.getenv("PERSIST_DIRECTORY"))
TOC_PERSIST_DIRECTORY = Path(os.getenv("TOC_PERSIST_DIRECTORY"))

# for LLM
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")
GROQ_MODEL            = os.getenv("GROQ_MODEL")   
OLLAMA_MODEL          = os.getenv("OLLAMA_MODEL")
MAX_TOKENS            = int(os.getenv("MAX_TOKENS"))   
OLLAMA_HOST_DEFAULT   = os.getenv("OLLAMA_HOST")

# 2. Check if a specific environment variable is set to indicate local execution
# If 'IS_LOCAL' is set to 'True', override the host to 'localhost'
if os.getenv("IS_LOCAL", "False").lower() == "true":
    OLLAMA_HOST = OLLAMA_HOST_DEFAULT.replace("ollama", "localhost")
else:
    OLLAMA_HOST = OLLAMA_HOST_DEFAULT