from config import DATA_TYPE, DATA_PATH, TOC_PATH
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser

#-------------------------------------------------------------------------------------------------------------------------------------------------
def documents_loader():
    """
        Loads and processes specified documents types from a specified file path.
        Returns:
            A list of processed documents ready for further use in a Langchain RAG pipeline.
    """
    print("Loading Books into documents")
    loader = GenericLoader(                       # GenericLoader - A generic document loader that allows combining \
                                                  # an arbitrary blob loader with a blob parser.

        blob_loader=FileSystemBlobLoader(         # Loads the file types found in the toc path (adjustable in the .env file).
            path=(DATA_PATH),                     # Env variable for the path of the files
            glob=(DATA_TYPE)                      # Env variable for the type of the files
        ),

        blob_parser=PyMuPDFParser(                # Reads and extracts content from PDF files.                      
            mode="single",                        # Single - takes all pages as a Langchain Document
                                                  # Page   - takes each pdf page as a Langchain Document
            extract_tables="markdown"             # Markdown(better for Ollama's/Groq's  LLMs to parse) or CSV, HTML formats
        )
    )
    # Storing the processed PDFs in a list of langchain documents.
    
    documents = []                                # Initializing the list
    documents.extend(loader.lazy_load())          # Adding processed PDFs to the list \
                                                  # by using lazy_load() to create an iterator for memory-efficient processing.
    print("Documents loaded")
    return documents
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
def documents_table_of_contents_loader():
    """
        Optional feature:
        Loading Table of Contents (toc) PDFs\
        Usage: Tool that answer to the query categories : 'What topics is the program trained on' \
              OR passed as a summary in the system prompt for the LLM.
    """
    print("Loading TOC into documents")
    loader = GenericLoader(                       # GenericLoader - A generic document loader that allows combining \
                                                  # an arbitrary blob loader with a blob parser.
 
        blob_loader=FileSystemBlobLoader(         # Loads the file types found in the toc path (adjustable in the .env file).
            path=(TOC_PATH),                      # Env variable for the path of the files
            glob=(DATA_TYPE)                      # Env variable for the type of the files
        ),

        blob_parser=PyMuPDFParser(                # Reads and extracts content from PDF files.                      
            mode="single",                        # Single - takes all pages as a Langchain Document
                                                  # Page   - takes each pdf page as a Langchain Document
            extract_tables="markdown"             # Markdown(better for Ollama's/Groq's  LLMs to parse) or CSV, HTML formats
        )
    )

    # Storing the processed PDFs in a list of langchain documents.

    table_of_contents = []                        # Initializing the list    
    table_of_contents.extend(loader.lazy_load())  # Adding processed PDFs to the list \
                                                  # by using lazy_load() to create an iterator for memory-efficient processing.
    print("TOC Documents loaded")
    return table_of_contents
#-------------------------------------------------------------------------------------------------------------------------------------------------