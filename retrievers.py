from vectorstores import documents_vectorstore, toc_vectorstore

#-------------------------------------------------------------------------------------------------------------------------------------------------
def documents_retriever():
    print("\nCreating book documents retriever.")
    documents_retriever = documents_vectorstore().as_retriever(
        search_type="mmr",      # specifies the search algorithm to use. MMR (Maximal Marginal Relevance) aims to find a diverse set of results.
        search_kwargs={"k": 2, "fetch_k": 10}  # sets the number of documents to retrieve to 4.
        )
    print("Book documents retriever loaded.")
    return documents_retriever
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
def toc_retriever():
    print("Creating TOC documents retriever.")
    toc_retriever = toc_vectorstore().as_retriever(
        search_type="mmr",      # specifies the search algorithm to use. MMR (Maximal Marginal Relevance) aims to find a diverse set of results.
        search_kwargs={"k": 20} # sets the number of documents to retrieve to 20 in order to retrieve the most of the table of contents.
        )
    print("TOC documents retriever loaded.")
    return toc_retriever
#-------------------------------------------------------------------------------------------------------------------------------------------------
