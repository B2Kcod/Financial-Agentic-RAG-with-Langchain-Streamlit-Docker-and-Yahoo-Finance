Description:

Finance Assistant:
A professional-grade financial assistant application built with Streamlit, LangChain, and yfinance, designed to provide insights into finance topics and real-time financial data. The application uses a Retrieval-Augmented Generation (RAG) pipeline to answer queries based on uploaded finance books and table of contents (TOC) PDFs, supplemented by real-time stock and ETF data from Yahoo Finance.

Features:

Finance Knowledge Base: Answers queries on finance topics by retrieving relevant information from uploaded PDF books.

Real-Time Financial Data: Fetches stock/ETF prices, financials, balance sheets, cash flow statements, and general company information using the Yahoo Finance API.

Conversational Memory: Maintains chat history with token-aware memory management to ensure efficient context retention within defined limits.

Dual LLM Support: Utilizes Groq as the primary language model with Ollama as a fallback, ensuring robustness in case of API limits or failures.

Streamlit UI: Provides an intuitive web interface for uploading PDFs, querying the assistant, and managing data, with a scrollable chat history.

Vectorstore Management: Automatically creates and persists vectorstores for efficient document retrieval using Chroma and HuggingFace embeddings.

Flexible Data Upload: Supports uploading finance books and optional TOC PDFs, with options to delete data and reset vectorstores.

Upload Documents:

Use the sidebar to upload finance-related PDF books to the DATA_PATH.

Upload TOC PDFs to the TOC_PATH for enhanced context retrieval.

Manage Data:

Use the "Delete All PDFs and Vectorstores" button in the sidebar to reset the application by clearing uploaded PDFs and vectorstores.

Query the Assistant:

Enter finance-related queries in the chat input field (e.g., "What is value investing?", "What is Tesla's stock price?", "Analyze Apple with technical analysis").

The assistant will use the document retrieval tool for finance topics, the Yahoo Finance tool for real-time data, or both for combined analysis queries.

Installation:

How to Build and Run It with Docker:

Step 0: Set up in .env your GROQ_API_KEY= 

Step 1: Build the Image (Choose GPU or CPU)

You need to run the build command once to create your application image.

GPU (NVIDIA with Cuda Toolkit version 12.8)	

docker-compose -f docker-compose.yml build

CPU (Universal)	

docker-compose -f docker-compose-cpu.yml build

Step 2: Run the Services

After a successful build, use the same file to start both the app and ollama services.

Target	Command

GPU (NVIDIA with Cuda Toolkit version 12.8)	

docker-compose -f docker-compose.yml up

CPU (Universal)

docker-compose -f docker-compose-cpu.yml up

The first time you run the up command, the ollama service will automatically pull the qwen2.5:latest model, which will be persisted to the ollama_models volume for faster startups in the future.

Once the services are running, access your application at: http://localhost:8000




