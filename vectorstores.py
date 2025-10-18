from config import PERSIST_DIRECTORY, TOC_PERSIST_DIRECTORY
import os
from loaders import documents_loader, documents_table_of_contents_loader
from splitter import documents_splitter
from llm_embedder_tokeziner import embedder
from langchain_chroma import Chroma

#-------------------------------------------------------------------------------------------------------------------------------------------------
def documents_vectorstore():
    """
    This function creates or loads a vector store for the book documents.
    """
    print("Creating or loading book documents vectorstore.")
    # If the directory doesn't exist or is empty, create a new vector store.
    if not (PERSIST_DIRECTORY.exists() and os.listdir(PERSIST_DIRECTORY)): 
        documents_vectorstore = Chroma.from_documents(
            documents=documents_splitter(documents_loader()), # Loads and splits the documents before adding them to the vector store.
            embedding=embedder,                               # Use the specified embedding model to convert splitted_documents to vectors.
            persist_directory= PERSIST_DIRECTORY              # Set the directory to save the vector store for future use.
            )
        print("Book documents vectorstore created.")
    # If the directory exists and contains data, load the existing vector store.
    else: 
        documents_vectorstore = Chroma( 
            embedding_function=embedder,        # Specify the embedding model that was used to create the vectors.
            persist_directory=PERSIST_DIRECTORY # Point to the directory where the vector store is saved.
            )
        print("Book documents vectorstore loaded.")
    return documents_vectorstore # Returns the created or loaded vectorstore.
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
def toc_vectorstore():
    """
    This function creates or loads a vector store for the table of contents (TOC) documents.
    """
    print("Creating or loading table of contents documents vectorstore.")
    # If the directory doesn't exist or is empty, create a new vector store.
    if not (TOC_PERSIST_DIRECTORY.exists() and os.listdir(TOC_PERSIST_DIRECTORY)): 
        toc_vectorstore = Chroma.from_documents(
            
            documents=documents_splitter(documents_table_of_contents_loader()), # Loads and splits the TOC documents before adding them to the\
                                                                                # vector store.

            embedding=embedder,                      # Use the specified embedding model to convert splitted_documents to vectors.
            persist_directory= TOC_PERSIST_DIRECTORY # Set the directory to save the vector store for future use.
            )
        print("Table of contents documents vectorstore created.")
    # If the directory exists and contains data, load the existing vector store.
    else:
        toc_vectorstore = Chroma(
            embedding_function=embedder,             # Specify the embedding model that was used to create the vectors.
            persist_directory= TOC_PERSIST_DIRECTORY # Point to the directory where the vector store is saved.
            )
        print("Table of contents documents vectorstore loaded.")
    return toc_vectorstore # Returns the created or loaded vectorstore.
#-------------------------------------------------------------------------------------------------------------------------------------------------