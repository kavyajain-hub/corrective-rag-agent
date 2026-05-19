from typing import Literal
# Import your Pydantic state schema
from src.state import RAGState 

def route_after_retrieval(state: RAGState) -> Literal["rewrite", "generate"]:
    """
    Evaluates the quality of retrieved documents based on the relevance score.
    
    If the document relevance score falls below the accepted threshold, it routes
    the state back to the query rewriter to optimize search keywords. Otherwise, 
    it advances safely to response generation.
    """
    print("\n[Edge Router] Evaluating Retrieval Quality...")
    
    # Define a strict threshold for content relevance
    RELEVANCE_THRESHOLD = 0.5
    
    if state.relevance_score < RELEVANCE_THRESHOLD:
        print(f"    ↳ Decision: REWRITE. Score ({state.relevance_score}) is below threshold ({RELEVANCE_THRESHOLD}).")
        return "rewrite"
        
    print(f"    ↳ Decision: GENERATE. Score ({state.relevance_score}) is sufficient.")
    return "generate"


def route_after_generation(state: RAGState) -> Literal["rewrite", "finalize"]:
    """
    Evaluates whether the generated response is structurally sound and grounded.
    
    If a hallucination flag is triggered during the auditing step, the system 
    rejects the draft and loops back to rewrite the query. If clean, the system
    proceeds to deliver the finalized answer.
    """
    print("\n[Edge Router] Auditing Draft Answer for Hallucinations...")
    
    if state.hallucination_flag:
        print("    ↳ Decision: REWRITE. Hallucination detected! Rerouting pipeline to self-correct.")
        return "rewrite"
        
    print("    ↳ Decision: FINALIZE. Draft answer verified as fully grounded in source context.")
    return "finalize"
