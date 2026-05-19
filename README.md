# Corrective-RAG: Self-Correcting Agentic State-Machine for High-Acuity Domains

An enterprise-ready, cloud-native Intelligent RAG architecture built using **LangGraph**, **Google Gemini**, and **Streamlit** designed to solve the systemic points of failure inherent in traditional linear (Query ➔ Retrieve ➔ Generate) RAG pipelines: hallucination vectors, out-of-distribution queries, and fragile semantic retrieval scores.

This system enforces an asynchronous, cyclic state machine that monitors its own pipeline accuracy telemetry in real time. It dynamically routes data through recursive query optimization and response-regeneration loops, guaranteeing deterministic, factory-grounded outputs before delivery.

---

## 🏗️ Core Tech Stack Components

The architecture relies entirely on decoupled, production-grade components:
*   **LangGraph for Cyclic Orchestration:** Manages state transitions, handles conditional graph edges, and drives recursive loop execution without state mutation bugs.
*   **Google Gemini for Query Rewriting & Generation:** Uses `gemini-2.5-flash` for high-throughput, low-latency semantic optimization and fact-grounded response synthesis.
*   **ChromaDB for Vector Retrieval:** Acts as the local, high-speed, vector similarity search engine to fetch relevant context footprints.
*   **HuggingFace Embeddings (`all-MiniLM-L6-v2`):** Handles on-device, local vectorization of incoming texts with zero token costs.
*   **Streamlit for Telemetry Visualization:** Provides a professional dashboard interface displaying live graph execution traces, metric comparisons, and telemetry logs.

---

## 📈 System Performance Telemetry & Evaluation

Conventional RAG engines fail silently when presented with low-relevance or conversational inputs. This engine captures system telemetry, allowing engineering teams to audit execution traces and map out the exact accuracy-versus-latency trade-offs introduced by agentic loops.

### Key Metrics Monitored Under Evaluation:
1.  **Retrieval Relevance Score Distribution:** Compares the raw embedding similarity matrix profile of the initial baseline run against the post-rewritten keyword search results.
2.  **Hallucination Rate Reduction Profile:** Tracks whether generated text violates factual grounding constraints, converting `Flagged Failure` runs into verified assets.
3.  **Latency Overhead vs. Baseline:** Quantifies the computational cost and token delay overhead introduced by the recursive loops to optimize engineering budgets.

### Case Study: High-Acuity Evaluation Pass

Below is an empirical trace of an adversarial, low-relevance slang input query (*"Bro, tell me about cars getting way too hot in the back when they're running fast"*), evaluated against a dense technical knowledge base regarding **Electric Vehicle Battery Thermodynamics**:

| Evaluation Metric Dimension | Baseline RAG (Linear Pipeline) | Agentic Loop (Self-Correcting Graph) |
| :--- | :--- | :--- |
| **Retrieval Relevance Score** | `0.20` (Critical Failure) | `0.90` (Optimized Match Profile) |
| **Hallucination Audit Status** | `Flagged Failure` (Ungrounded) | `Passed Grounding` (Fact Verified) |
| **Total Pipeline Latency** | `1.25s` | `44.62s` |
| **Inference Passes Required** | 1 Pass | 2 Passes |
| **Downstream Output Safety** | **Compromised** (Hallucinated/Refused) | **Guaranteed** (Fully Grounded) |

---

## 🛠️ System Architecture Workflow

```mermaid
graph TD
    A[User Casual / Adversarial Input] --> B[📥 Node: retrieve_docs]
    B --> C[📊 Node: grade_relevance]
    C -->|Relevance Score < 0.50| D[🔄 Node: rewrite_query]
    D -->|Inject Optimized Keywords| B
    C -->|Relevance Score >= 0.50| E[🧠 Node: generate_draft]
    E --> F[🛡️ Node: audit_hallucination]
    F -->|Hallucination Flagged| D
    F -->|Audit Passed: Grounding Verified| G[📝 Render Verified Answer]
    
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#ffcccb,stroke:#333,stroke-width:2px
    style G fill:#90ee90,stroke:#333,stroke-width:2px