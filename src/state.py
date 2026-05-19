import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RAGState(BaseModel):
    # Core Pipeline Data
    query: str
    rewritten_query: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = []
    draft_answer: Optional[str] = None
    final_answer: Optional[str] = None
    
    # Telemetry & Evaluation Metrics Data
    relevance_score: float = 0.0
    hallucination_flag: bool = False
    loop_count: int = 0
    
    # Metrics Trackers (For Baseline comparison)
    start_time: float = Field(default_factory=time.time)
    initial_relevance_score: Optional[float] = None
    initial_hallucination_flag: Optional[bool] = None
    total_latency: float = 0.0