from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import your custom workflow blocks
from src.state import RAGState
from src.nodes import (
    query_rewriter_node,
    retriever_node,
    relevance_grader_node,
    generator_node,
    hallucination_grader_node
)
from src.edges import (
    route_after_retrieval,
    route_after_generation
)

def compile_rag_graph(llm, vector_db):
    """
    Constructs, wires, and compiles the Self-Correcting Agentic RAG state machine.
    
    Injects required dependencies (LLM instance and Vector Database) directly
    into the processing nodes using partial functions.
    """
    # 1. Initialize StateGraph with your Pydantic BaseModel schema
    workflow = StateGraph(RAGState)

    # 2. Add Processing Nodes (injecting dependencies via functools.partial)
    workflow.add_node("rewrite_query", partial(query_rewriter_node, llm=llm))
    workflow.add_node("retrieve_docs", partial(retriever_node, vector_db=vector_db))
    workflow.add_node("grade_relevance", relevance_grader_node)
    workflow.add_node("generate_draft", partial(generator_node, llm=llm))
    workflow.add_node("audit_hallucination", hallucination_grader_node)

    # 3. Define Structural Execution Pathways
    # Start execution loop at document retrieval
    workflow.set_entry_point("retrieve_docs")

    # Wire sequential node edges
    workflow.add_edge("rewrite_query", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "grade_relevance")
    workflow.add_edge("generate_draft", "audit_hallucination")

    # 4. Integrate Conditional Routing Edges
    workflow.add_conditional_edges(
        "grade_relevance",
        route_after_retrieval,
        {
            "rewrite": "rewrite_query",  # Loop back if relevance score is low
            "generate": "generate_draft" # Pass to generation if score is sufficient
        }
    )

    workflow.add_conditional_edges(
        "audit_hallucination",
        route_after_generation,
        {
            "rewrite": "rewrite_query", # Loop back to self-correct if hallucination is flagged
            "finalize": END             # Exit graph cleanly if verification passes
        }
    )

    # 5. Compile the State Machine workflow
    compiled_app = workflow.compile(checkpointer=MemorySaver())
    return compiled_app