import streamlit as st
import os
import shutil
from config import DATA_PATH, TOC_PATH, PERSIST_DIRECTORY, TOC_PERSIST_DIRECTORY
from agent import run_finance_agent

#-------------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Finance Assistant", page_icon="📊", layout="wide")

st.title("💹Finance Assistant")

# 'messages' stores the chat history (user queries and assistant responses)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am your Finance Assistant. Ask me a question related to finance, and I'll use my knowledge base and tools to provide an answer.", "tokens": None}
    ]
# 'processing' tracks if the agent is currently generating a response
if "processing" not in st.session_state:
    st.session_state["processing"] = False
# 'last_query' stores the query that triggered the current processing
if 'last_query' not in st.session_state:
    st.session_state['last_query'] = None
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Data Management")
    
    st.subheader("Upload Books")
    uploaded_books = st.file_uploader("Select PDF books from your PC", type="pdf", accept_multiple_files=True, key="books_uploader")
    if uploaded_books:
        for uploaded_file in uploaded_books:
            file_path = os.path.join(DATA_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Books uploaded successfully! Vectorstore will be created automatically on first query if needed.")
    
    st.subheader("Upload Table of Contents (Optional)")
    uploaded_toc = st.file_uploader("Select TOC PDFs from your PC", type="pdf", accept_multiple_files=True, key="toc_uploader")
    if uploaded_toc:
        for uploaded_file in uploaded_toc:
            file_path = os.path.join(TOC_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("TOC uploaded successfully! TOC vectorstore will be created automatically on first query if needed.")
    
    if st.button("Delete All PDFs and Vectorstores"):
        # Delete PDFs from data and TOC paths
        for path in [DATA_PATH, TOC_PATH]:
            if os.path.exists(path):
                for filename in os.listdir(path):
                    if filename.endswith(".pdf"):
                        os.remove(os.path.join(path, filename))
        
        # Delete vectorstore directories
        for dir_path in [PERSIST_DIRECTORY, TOC_PERSIST_DIRECTORY]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        
        st.success("All PDFs and Vectorstores deleted successfully! Please refresh the application to fully reset.")
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
def process_query(query):
    """Handles the query submission, runs the agent, and updates chat history."""
    
    # Run the agent and get the final output and token count
    try:
        # 3. Run the agent and get the final output and token count
        # The agent.py function returns (final_output, total_tokens)
        with st.spinner('Thinking... The agent is processing your request and running tools...'):
            final_output, total_tokens = run_finance_agent(query)
        
        # 4. Append assistant message with output and token count to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_output, 
            "tokens": total_tokens
        })
        
    except Exception as e:
        error_message = f"An error occurred: {e}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error while processing your request.", "tokens": None})
    finally:
        # 5. Re-enable input by setting processing state to False and rerunning
        st.session_state["processing"] = False
        st.session_state['last_query'] = None # Clear the last query
        st.rerun()
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Use a fixed-height container to make the chat history scrollable
chat_container = st.container(height=550, border=True)

with chat_container:
    for message in st.session_state.messages:
        # Display the chat message using st.chat_message for a conversational look
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display token usage only for assistant messages that have a token count
            if message["role"] == "assistant" and message["tokens"] is not None:
                st.caption(f"Token Usage: {message['tokens']} tokens")

# Check if processing is active to disable the input field
input_disabled = st.session_state["processing"]

# Use st.chat_input, which natively handles the Enter key submission and displays the small arrow button
if query_input := st.chat_input(
    "Ask your finance question here...", 
    disabled=input_disabled,
    key="chat_input_field"
):
    # This block executes when the user submits a non-empty query
    
    # 1. Append user message to history
    st.session_state.messages.append({"role": "user", "content": query_input, "tokens": None})
    
    # 2. Temporarily disable input and store the query
    st.session_state["processing"] = True
    st.session_state['last_query'] = query_input
    
    # 3. Rerun to show the user's new message and disable the input field
    st.rerun()

# This block runs only when processing is True and we have a query stored
if st.session_state["processing"] and st.session_state['last_query']:
    process_query(st.session_state['last_query'])
#-------------------------------------------------------------------------------------------------------------------------------------------------