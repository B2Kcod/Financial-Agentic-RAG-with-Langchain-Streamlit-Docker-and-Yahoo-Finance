from config import MAX_TOKENS
from llm_embedder_tokenizer import groq_llm, ollama_llm, encoder
from tools import document_retrieval_tool, toc_retrieval_tool, yahoo_finance_tool
from prompts import ollama_agent_prompt, groq_agent_prompt
from langchain.agents import create_tool_calling_agent, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama 
from llm_embedder_tokenizer import OLLAMA_MODEL, OLLAMA_HOST 

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Initialize conversation memory to store chat history and manage token limits
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
def run_finance_agent(query: str):
    """ Run the finance agent with Groq LLM as primary and Ollama LLM as fallback primarely when Groq daily rimit is reached."""
    # Groq Agent
    try:
        tools = [document_retrieval_tool, yahoo_finance_tool, toc_retrieval_tool]
        llm = groq_llm()
        agent = create_react_agent(llm, tools, groq_agent_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory,
            return_intermediate_steps=True,
            max_iterations=3,
        )
        result = agent_executor.invoke({"input": query})
        final_output = result["output"]
        # Save the conversation context to memory whitin max token limits
        memory.save_context({"input": query}, {"output": final_output})
        # Reset memory if token limit exceeded
        total_tokens = sum(len(encoder.encode(msg.content)) for msg in memory.chat_memory.messages)
        if total_tokens > MAX_TOKENS:
            memory.clear()
            final_output += "\n\n⚠️ Memory reset: Token limit exceeded."

        return final_output, total_tokens
    # Fallback to Ollama if Groq fails
    except Exception as e:
        print(f"Grok LLM failed: {str(e)}. Falling back to Ollama LLM.")
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
        tools = [document_retrieval_tool, yahoo_finance_tool]
        toc_output = ""
        toc_output = toc_retrieval_tool.invoke("List everything")
        
        # Considering Ollama Agent connection for streamlit deployment and docker compatibility we have to implement\
        #  a two-step fallback logic:
        # Ollama Attempt 1: Default Host (as per config)
        try:
            llm = ollama_llm()
            agent = create_tool_calling_agent(llm, tools, ollama_agent_prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                memory=memory,
                return_intermediate_steps=True,
                max_iterations=3,
            )

            result = agent_executor.invoke({"input": query})
            print("Successfully connected to Ollama on default host.")

        # Ollama Attempt 2: Localhost Fallback
        except Exception as ollama_error:
            print(f"Ollama LLM failed on default host: {str(ollama_error)}. Retrying with localhost.")

            try:
                llm = ChatOllama(
                        model=OLLAMA_MODEL,
                        temperature=0,    
                        base_url=OLLAMA_HOST.replace("ollama", "localhost"), 
                        timeout=30
                    )
                agent = create_tool_calling_agent(llm, tools, ollama_agent_prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    memory=memory,
                    return_intermediate_steps=True,
                    max_iterations=3,
                )
                result = agent_executor.invoke({"input": query})
                print("Successfully connected to Ollama using localhost fallback.")

            except Exception as final_error:
                print(f"Ollama LLM failed on localhost: {str(final_error)}. Agent failed completely.")
                return "Sorry, both the primary (Groq) and fallback (Ollama) language models are currently unavailable.", 0
#-------------------------------------------------------------------------------------------------------------------------------------------------
        # Ollama agent is unable to follow ReAct or Tool-Calling properly so we need to synthesize the final output manually:
        document_output = ""
        yahoo_output = ""
    
        for step in result["intermediate_steps"]:
            if isinstance(step, tuple) and len(step) == 2 and step[0].tool == "yahoo_finance_tool":
                yahoo_output = str(step[1])
            if isinstance(step, tuple) and len(step) == 2 and step[0].tool == "document_retrieval_tool":
                document_output = str(step[1])

        synthesis_prompt = f"""
        You are a professional financial assistant.
        YOU MUST ONLY SPEAK ABOUT THE FOLLOWING #### Subjects #####:
        #### Subjects ##### 
        {toc_output}
        #### Subjects ##### 
        You also have the ability to gather realtime data of stocks/etfs and other tradable assets.

        Strict Rules (NO EXCEPTIONS - Follow these step-by-step in your reasoning):
        1. First, evaluate the Original Query: Check if it directly relates to the #### Subjects ##### above or gathering financial data (e.g., stocks/ETFs). If not (e.g., if about insurance, health, or unrelated topics), DO NOT use any context. Instead, politely redirect to the topics in #### Subjects #####.
        2. If the query is related, respond ONLY based on what's between ####Context####. Do NOT add external knowledge, assumptions, or pretrained information.
        3. Respond in the language the user is asking or requesting, even though your reasoning and received data below is in English.
        4. No jokes, no casual conversation, no extra commentary!
        5. Do not display your reasoning/logic in your final answer! Respond directly! The user doesn't need to see your Rules, Step-by-Step Reasoning, examples or the ##Context#### or #### Subjects ##### is above.

        Step-by-Step Reasoning (Do this internally, do NOT show in output):
        - Step 1: Read #### Subjects #####. List key topics: [internally list them].
        - Step 2: Compare Original Query to subjects. Is it a match? Yes/No.
        - Step 3: If No match, prepare redirection: "I'm specialized in [brief list of subjects]. For your query on [topic], please ask about [subjects]."
        - Step 4: If Yes, synthesize a concise response using ONLY ####Context####.
        - Step 5: Ensure response is professional and direct.

        Examples:
        - Query: "What is car insurance?" (Not in subjects) → Response: "I'm specialized in finance topics like stocks, trading strategies, and ETFs. For insurance queries, please redirect to those areas."
        - Query: "What is Tesla's stock price?" (Related to financial data) → Response: [Synthesize from yahoo_output only, e.g., "Tesla (TSLA) current price is $250. Volume: 100M."]
        - Query: "Explain value investing." (If in subjects and context) → Response: [Synthesize from document_output only].

        Original Query: {query}

        ####Context####
        {yahoo_output}
        {document_output}
        ####Context####
        """

        synthesis_llm = llm 
        synthesis_result = synthesis_llm.invoke(synthesis_prompt)
        final_output = synthesis_result.content

        # Save the conversation context to memory whitin max token limits
        memory.save_context({"input": query}, {"output": final_output})
        # Reset memory if token limit exceeded
        total_tokens = sum(len(encoder.encode(msg.content)) for msg in memory.chat_memory.messages)
        if total_tokens > MAX_TOKENS:
            memory.clear()
            final_output += "\n\n⚠️ Memory reset: Token limit exceeded."

        return final_output, total_tokens
#-------------------------------------------------------------------------------------------------------------------------------------------------