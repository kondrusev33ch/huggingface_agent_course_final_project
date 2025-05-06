import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph, MessagesState
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from supabase.client import create_client

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a - b


@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a % b 


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}


@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}


@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}



class BasicAgent:
    """A langgraph agent."""
    def __init__(self):
        self.graph = self.build_graph()
        print("BasicAgent initialized.")

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        # Wrap the question in a HumanMessage from langchain_core
        messages = [HumanMessage(content=question)]
        messages = self.graph.invoke({"messages": messages})
        answer = messages['messages'][-1].content
        return answer[14:]
    
    def create_system_message(self, file_path: str="system_prompt.txt") -> str:
        # Load the system prompt
        with open(file_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

        return SystemMessage(content=system_prompt)
    
    def create_vector_store() -> SupabaseVectorStore:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #  dim=768
        supabase = create_client(os.environ.get("SUPABASE_URL"), 
                                 os.environ.get("SUPABASE_SERVICE_KEY"))
        return SupabaseVectorStore(client=supabase,
                                   embedding= embeddings,
                                   table_name="documents",
                                   query_name="match_documents_langchain")
    
    def build_graph(self) -> StateGraph:
        """Build the graph"""
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620",    # or "claude-3-7-sonnet-latest"
                            temperature=0)  # if needed: max_tokens=4096, timeout=60

        # Tools
        tools = [multiply,
                 add,
                 subtract,
                 divide,
                 modulus,
                 wiki_search,
                 web_search,
                 arvix_search]

        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)

        # Vector store
        vector_store = self.create_vector_store()

        # Node
        def assistant(state: MessagesState):
            """Assistant node"""
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
                     
        def retriever(state: MessagesState):
            """Retriever node"""
            similar_question = vector_store.similarity_search(state["messages"][0].content)
            example_msg = HumanMessage(
                content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
            )
            return {"messages": [self.create_system_message()] + state["messages"] + [example_msg]}

        builder = StateGraph(MessagesState)
        builder.add_node("retriever", retriever)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "retriever")
        builder.add_edge("retriever", "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")

        # Compile graph
        return builder.compile()


if __name__ == "__main__":
    pass