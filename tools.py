import yfinance as yf
from retrievers import documents_retriever,toc_retriever
from langchain_core.tools import tool

#-------------------------------------------------------------------------------------------------------------------------------------------------
@tool
def document_retrieval_tool(query: str) -> str:
    """
    Use this tool for queries that refer to finance domain.
    Examples:
    - What is ... strategy/analyses?
    - Analyze ...
    - What do you think about...
    """
    retriever = documents_retriever()
    response = retriever.invoke(query)
    formatted = "\n\n".join([doc.page_content for doc in response])  # Combines only the page content into single string
    return formatted
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
@tool
def toc_retrieval_tool(query: str) -> str:
    """
    Use this tool for queries that refer to your general knowledge\
    (Examples: - What ... you ... trained on?
               - What are ... topics/chapters?)
    """
    retriever = toc_retriever()         # Initializing the TOC retriever.
    response = retriever.invoke(query)  # Invoke the TOC retriever with the user's query to fetch relevant documents.
    formatted = "\n\n".join([doc.page_content for doc in response])  # Combines only the page content into single string
    return formatted
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
@tool
def yahoo_finance_tool(input_str: str) -> str:
    """
    Use this tool for financial data, prices, volumes, balance sheets, cash_flow and general information about anay stocks/etf's/ or tradable asssets.
    - Always convert company names to ticker symbols before calling (e.g., Tesla → TSLA, Apple → AAPL).
    - Input format: "<ticker> <command> [param1=value1] [param2=value2] ..."
      Supported commands:
        - history: Get historical stock/ETF data. Params: period (e.g., 1mo, 6mo, 1y), interval (e.g., 1d, 1wk).
        - info: General company information.
        - financials: Income statement data.
        - balance_sheet: Balance sheet data.
        - cash_flow: Cash flow statement.
        - quote: Current quote and stats.
    """
    # Split the input string into parts based on whitespace
    parts = input_str.split()

    # Check if the input string is empty or lacks a ticker symbol
    if len(parts) < 1:
        return "Invalid input: Missing ticker symbol."

    # Extract the ticker symbol (first part of the input)
    ticker = parts[0]

    # Set the command to the second part of the input, default to "info" if not provided
    command = parts[1] if len(parts) > 1 else "info"

    # Initialize an empty dictionary to store optional parameters
    params = {}

    # Parse additional parameters (key=value pairs) from the remaining parts
    for p in parts[2:]:
        if '=' in p:
            # Split each parameter into key and value (using the first '=' as delimiter)
            key, value = p.split('=', 1)
            # Store the key-value pair in the params dictionary
            params[key] = value

    try:
        # Create a yfinance Ticker object for the given stock ticker
        stock = yf.Ticker(ticker)
        
        # Handle the "history" command to fetch historical stock data
        if command == "history":
            period = params.get("period", "1mo")
            interval = params.get("interval", "1d")
            hist = stock.history(period=period, interval=interval)
            if not hist.empty:
                # Format as a readable string table
                result = f"Historical data for {ticker} (period={period}, interval={interval}):\n"
                result += "Date                  Open      High      Low       Close     Volume\n"
                for date, row in hist.iterrows():
                    result += f"{date.strftime('%Y-%m-%d %H:%M:%S'):20} {row['Open']:8.2f} {row['High']:8.2f} {row['Low']:8.2f} {row['Close']:8.2f} {row['Volume']:10.0f}\n"
                return result
            return f"No historical data found for {ticker}."

        # Handle the "info" command to fetch general stock information
        elif command == "info":
            info = stock.info
            result = f"Company Information for {ticker}:\n"
            for key, value in info.items():
                result += f"{key}: {value}\n"
            return result
        
        # Handle the "financials" command to fetch income statement data
        elif command == "financials":
            if hasattr(stock, 'income_stmt'):
                financials = stock.income_stmt
                result = f"Income Statement for {ticker}:\n"
                for date, row in financials.items():
                    result += f"Date: {date}\n"
                    for key, value in row.items():
                        result += f"  {key}: {value}\n"
                return result
            return f"No financials data available for {ticker}."

        # Handle the "balance_sheet" command to fetch balance sheet data
        elif command == "balance_sheet":
            if hasattr(stock, 'balance_sheet'):
                balance = stock.balance_sheet
                result = f"Balance Sheet for {ticker}:\n"
                for date, row in balance.items():
                    result += f"Date: {date}\n"
                    for key, value in row.items():
                        result += f"  {key}: {value}\n"
                return result
            return f"No balance sheet data available for {ticker}."

        # Handle the "cash_flow" command to fetch cash flow data
        elif command == "cash_flow":
            if hasattr(stock, 'cashflow'):
                cashflow = stock.cashflow
                result = f"Cash Flow Statement for {ticker}:\n"
                for date, row in cashflow.items():
                    result += f"Date: {date}\n"
                    for key, value in row.items():
                        result += f"  {key}: {value}\n"
                return result
            return f"No cash flow data available for {ticker}."

        # Handle the "quote" command to fetch current stock quote
        elif command == "quote":
            if hasattr(stock, 'fast_info'):
                quote = dict(stock.fast_info)
                result = f"Current Quote for {ticker}:\n"
                for key, value in quote.items():
                    result += f"{key}: {value}\n"
                return result
            else:
                price = stock.info.get('regularMarketPrice', 'No quote available.')
                return f"Current Quote for {ticker}: regularMarketPrice: {price}"

        # Handle unknown commands
        else:
            return f"Unknown command: {command}."

    # Catch any exceptions that occur during data fetching
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"
#-------------------------------------------------------------------------------------------------------------------------------------------------