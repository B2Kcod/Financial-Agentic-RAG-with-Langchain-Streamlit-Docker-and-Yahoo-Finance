from config import ENCODER_NAME, CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm_embedder_tokenizer import encoder
import warnings

#-------------------------------------------------------------------------------------------------------------------------------------------------
def documents_splitter(documents):
    """
    Splits loaded documents into smaller chunks using a token aware recursive text splitter
    Returns:
        list: A list of Document objects, where each object represents a text chunk.
    """
    print("Spltting documents for embedder")
    # Initialize the document splitter using a tokenizer-aware approach.
    documents_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(

        encoding_name=ENCODER_NAME,          # Encoder must be compatible with the embedder.
        chunk_size=CHUNK_SIZE,               # Maximum token limit of the embedding model. 
        chunk_overlap=CHUNK_OVERLAP,         # Overlaping chunks to retain context.
        separators=SEPARATORS                # Attempts to split in order, by separators.
    )

    # Loading documents from loaders.py module and splitting them into chunks.
    documents_splits = documents_splitter.split_documents(documents)

    # Checking the token count of each split to ensure it doesn't exceed the embedder's limit.
    for i, split in enumerate(documents_splits):
        token_count = len(encoder.encode(split.page_content))
        if token_count > CHUNK_SIZE:
            warnings.warn(
                f"Split {i} exceeds the maximum token limit of 1024. Token count: {token_count}",
                UserWarning
            )
    print("Documents splitted")
    return documents_splits
#-------------------------------------------------------------------------------------------------------------------------------------------------