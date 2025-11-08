from config import ENCODER_NAME, EMBEDDER_NAME, GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL, OLLAMA_HOST
import tiktoken
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

#-------------------------------------------------------------------------------------------------------------------------------------------------
encoder = tiktoken.get_encoding(ENCODER_NAME)          # Initializing the encoder in order to make the splitter split by tokens not by characters.
                                                       # Encoder must be compatible with the embedder.
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Cuda uses the more efficient GPU VRAM for embeddings but it requires Nvidia Graphics Card and Cuda Toolkit software with the same version \
# as pytorch library: For a system with Nvidia Graphics Card and 12.8 Cuda Toolkit software: \
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

device = "cuda" if torch.cuda.is_available() else "cpu" # If the conditions aren't met, it switches to using CPU for embeddings.

embedder = HuggingFaceEmbeddings(
    model_name=EMBEDDER_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}        # Normalize embeddings to a length of 1 for effective cosine similarity search.
)
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
def groq_llm():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,   
        temperature=0         # Sets the randomness of the response from 0-1. (0 - no randomness)
    )
    return llm
def ollama_llm():
    llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,    # Sets the randomness of the response from 0-1. (0 - no randomness) 
            base_url=OLLAMA_HOST,
            timeout=30
        )
    return llm
#-------------------------------------------------------------------------------------------------------------------------------------------------