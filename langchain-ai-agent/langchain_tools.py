from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from datetime import datetime


# DuckDuckGo tool
ddg_search = DuckDuckGoSearchResults()

@tool
def ddg_tool(query: str) -> str:
    """
    Search the web for real-time information and news. 
    """
    return ddg_search.run(query)

# Wikipedia tool
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia_search = WikipediaQueryRun(api_wrapper=wiki_api)

@tool
def wikipedia_tool(query: str) -> str:
    """
    Search Wikipedia for historical information. 
    """
    return wikipedia_search.run(query)



@tool
def txt_file_tool(llm_output: str, filename: str = None) -> str:
    """
    REQUIRED STEP: After completing the search with either DDG or Wikipedia (or both), please save the structured output into a text (.txt) file. Make sure the name of this ouput file has the filename defined in the function argument. 

     Arguments:
        llm_output: The research content to save
        filename: If not provided, uses timestamp.
    
    Returns:
        Success or error message
    """

    timestamp = datetime.now().strftime("%Y-%m-%d")

    if not filename.endswith('.txt'):
        filename = f"ai-research-output-{timestamp}.txt"

    research_text = (
        f"--- Research Entry: {timestamp} ---\n"
        f"{llm_output}"
    )

    try:
        with open(file=f"{filename}", mode='a', encoding="UTF-8") as file:
            file.write(research_text)
        return f"File saved to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"