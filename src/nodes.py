from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.state import RAGState
import time

def query_rewriter_node(state: RAGState, llm) -> Dict[str, Any]:
    """
    Optimizes the user query into precise technical keywords designed 
    to maximize vector database search accuracy.
    """
    print("\n--- [NODE] REWRITING USER QUERY ---")
    
    system_prompt = (
        "You are an expert search-query optimizer for Vector Databases.\n"
        "Your task is to analyze an underperforming user query and rewrite it to focus "
        "strictly on core semantic keywords, technical terms, and concepts.\n"
        "Remove conversational fluff and question words.\n\n"
        "Original Query: {query}"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | llm
    
    response = chain.invoke({"query": state.query})
    optimized_query = response.content.strip()
    
    print(f"    ↳ [Old Query]: {state.query}")
    print(f"    ↳ [New Query]: {optimized_query}")
    
    return {"rewritten_query": optimized_query}


def retriever_node(state: RAGState, vector_db) -> Dict[str, Any]:
    """
    Queries ChromaDB using either the optimized rewritten query (if available) 
    or the original query as a fallback. Serializes output chunks into dictionaries.
    """
    print("\n--- [NODE] RETRIEVING CHUNKS FROM CHROME DB ---")
    
    # Route search term dynamically based on what's available in state
    search_query = state.rewritten_query if state.rewritten_query else state.query
    print(f"    ↳ Running vector similarity search for: \"{search_query}\"")
    
    results = vector_db.similarity_search(search_query, k=5)
    
    # Format standard LangChain Document objects into clean, JSON-serializable dictionaries
    formatted_chunks = [
        {"text": doc.page_content, "metadata": doc.metadata} 
        for doc in results
    ]
    
    print(f"    ↳ Successfully retrieved {len(formatted_chunks)} context chunks.")
    return {"retrieved_chunks": formatted_chunks}


def relevance_grader_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- [NODE] GRADING DOCUMENT RELEVANCE ---")
    
    # Dynamic heuristic based on mock terms or actual token overlaps
    has_keywords = any(term in state.query.lower() or (state.rewritten_query and term in state.rewritten_query.lower()) 
                       for term in ["thermal", "runaway", "battery", "health", "degradation"])
    
    score = 0.9 if (len(state.retrieved_chunks) > 0 and has_keywords) else 0.2
    
    updates: Dict[str, Any] = {"relevance_score": score}
    
    # Capture baseline metric if this is the first iteration pass
    if state.initial_relevance_score is None:
        updates["initial_relevance_score"] = score
        
    return updates


def generator_node(state: RAGState, llm) -> Dict[str, Any]:
    """
    Synthesizes a draft response strictly grounded in the retrieved context chunks.
    """
    print("\n--- [NODE] GENERATING DRAFT ANSWER ---")
    
    # Flatten array of chunk dicts into a single context string
    context_str = "\n\n".join([chunk["text"] for chunk in state.retrieved_chunks])
    
    system_prompt = (
        "Answer the user query completely and accurately using only the provided context snippets.\n"
        "If you do not know the answer based on the context, state that you do not know.\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | llm
    
    response = chain.invoke({
        "context": context_str,
        "query": state.query
    })
    
    return {"draft_answer": response.content}


def hallucination_grader_node(state: RAGState) -> Dict[str, Any]:
    print("\n--- [NODE] AUDITING ANSWER FOR HALLUCINATIONS ---")
    
    # Simple evaluation rule simulation: triggers a hallucination flag 
    # on the first run if specific grounding sources aren't mentioned
    if state.loop_count == 0 and "source" not in state.draft_answer.lower():
        is_hallucinated = True
    else:
        is_hallucinated = False
        
    updates: Dict[str, Any] = {
        "hallucination_flag": is_hallucinated,
        "loop_count": state.loop_count + 1
    }
    
    # Capture baseline hallucination rate profile
    if state.initial_hallucination_flag is None:
        updates["initial_hallucination_flag"] = is_hallucinated
        
    if not is_hallucinated:
        updates["final_answer"] = state.draft_answer
        updates["total_latency"] = time.time() - state.start_time
        
    return updates