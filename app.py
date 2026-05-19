import os
import time
import uuid
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.graph import compile_rag_graph
from src.state import RAGState

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Self-Correcting RAG Telemetry", 
    page_icon="📈", 
    layout="wide"
)

def create_llm():
    """Initializes cloud-native Google Gemini model for fast, agentic reasoning."""
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing `GOOGLE_API_KEY` environment variable. Please retrieve one from Google AI Studio.")
        st.stop()
    
    # We use gemini-2.5-flash as the default choice due to ultra-low latency profiles
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)


@st.cache_resource
def initialize_vector_db():
    """
    Dynamically loads raw unstructured files from the data directory,
    chunks them for efficient context injection, and creates local vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    data_dir = "./data"
    
    # Create directory if it doesn't exist yet
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.warning(f"Created empty `{data_dir}/` directory. Please add text files there.")
        return None

    # Load all text documents inside the directory
    loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
    raw_documents = loader.load()
    
    if not raw_documents:
        st.error("No `.txt` files found in the `data/` folder. Please add raw knowledge data files.")
        st.stop()
        
    # Split documents into small semantic chunks (essential for clean RAG grounding)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    semantic_chunks = text_splitter.split_documents(raw_documents)
    
    # Store natively into local Chroma database instances
    vector_db = Chroma.from_documents(
        documents=semantic_chunks, 
        embedding=embeddings
    )
    return vector_db


# --- CORE STRUCTURAL INITIALIZATION ---
llm = create_llm()
vector_db = initialize_vector_db()
rag_agent = compile_rag_graph(llm=llm, vector_db=vector_db)

st.title("📈 Self-Correcting RAG Performance Dashboard (Gemini)")

user_query = st.text_input("Run Pipeline Query:", placeholder="e.g., Explain battery heat failure mechanisms?")

if user_query:
    # Generate a fresh, unique transaction thread ID for each query submission
    unique_thread_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": unique_thread_id}}
    
    initial_inputs = RAGState(query=user_query).model_dump()
    
    col_logs, col_metrics = st.columns([4, 3])
    
    with col_logs:
        st.subheader("⚙️ Graph Execution Trace Log")
        with st.status("Tracing Graph Nodes...", expanded=True) as status:
            for output in rag_agent.stream(initial_inputs, config=thread_config, stream_mode="updates"):
                for node_name, delta in output.items():
                    st.write(f"✔️ Completed node **{node_name}**")
            status.update(label="Graph complete!", state="complete")
            
    # Pull terminal execution values from graph memory storage using current thread ID
    state_history = rag_agent.get_state(thread_config).values
    
    # Calculate Latency Overhead Simulation metrics
    baseline_latency = 1.25  # Estimated baseline time a single execution path takes
    actual_latency = state_history.get("total_latency", baseline_latency)
    latency_overhead = max(0.0, actual_latency - baseline_latency)

    with col_metrics:
        st.subheader("📊 Agent Loop vs. Baseline Metrics")
        
        # 1. Relevance Score Component
        init_rel = state_history.get("initial_relevance_score", 0.0)
        final_rel = state_history.get("relevance_score", 0.0)
        rel_delta = final_rel - init_rel
        
        # 2. Hallucination Rates Component
        init_hal = "Flagged Failure" if state_history.get("initial_hallucination_flag") else "Passed Grounding"
        final_hal = "Flagged Failure" if state_history.get("hallucination_flag") else "Passed Grounding"
        
        # Draw cleanly onto visual columns layout grid
        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                label="Retrieval Relevance Score", 
                value=f"{final_rel:.2f}", 
                delta=f"+{rel_delta:.2f}" if rel_delta > 0 else f"{rel_delta:.2f}"
            )
        with m2:
            st.metric(
                label="Total Network Latency", 
                value=f"{actual_latency:.2f}s", 
                delta=f"+{latency_overhead:.2f}s Overhead", 
                delta_color="inverse"
            )
            
        st.write("---")
        
        # Performance Comparison Table
        metrics_table = {
            "Evaluation Metric Dimension": ["Relevance Assessment", "Hallucination Audit", "Inference Passes Required"],
            "Baseline RAG (Linear)": [f"{init_rel:.2f}", "Flagged Failure" if state_history.get("initial_hallucination_flag") else "Passed Grounding", "1 Pass"],
            "Agentic Loop (Self-Correcting)": [f"{final_rel:.2f}", final_hal, f"{state_history.get('loop_count', 1)} Passes"]
        }
        st.table(metrics_table)
        
        # Final Output Display
        st.subheader("📝 Verified Answer Output")
        st.info(state_history.get("final_answer", "No valid answer generated."))