## Scaling & Orchestration: Reliability at Scale

This document outlines the orchestration logic, error-handling resilience, and future scaling roadmap for the system. By using a graph-based state machine rather than a linear script, the system achieves operational robustness suitable for production environments.

---

### 1. Orchestration Graph
The system utilizes a **state-machine-based orchestration graph**. Each request flows through independent nodes that operate on a shared `OrchestrationState`, allowing for complex branching and parallel execution.

#### Execution Flow
The following sequence defines the lifecycle of a single `/chat` request:

```text
POST /chat
  ↓
intent_classification
  ↓
(memory_retrieval || document_retrieval)  ← [Parallel I/O]
  ↓
tool_planning
  ↓
tool_execution
  ↓
response_generation
  ↓
memory_write_back                        ← [Post-Response Persistence]
```

**Key Design Properties:**
* **Shared State:** All nodes mutate a single `OrchestrationState` object, ensuring a single source of truth.
* **Parallel Retrieval:** Memory (L2/L3) and RAG (L4) lookups run concurrently to minimize wall-clock latency.
* **Isolation:** Each node is independent; a failure in the tool-execution layer does not necessarily prevent the system from generating a response based on retrieved documents.
* **Observability:** Each node emits trace events with precise timing and metadata for performance profiling.

---

### 2. Failure-Mode Matrix
Production systems must "fail softly." The following matrix describes how Chronos mitigates risks across the pipeline:

| Component | Failure Type | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **Intent Classification** | Misclassification | Wrong routing | Rule-based signals + LLM fallback |
| **Memory Retrieval** | SQLite read failure | Missing personalization | Safe fallback (proceed with empty memory) |
| **Document Retrieval** | Vector DB timeout | No grounding | Keyword fallback + proceed with empty context |
| **Tool Execution** | API crash / error | Partial answer | Isolated execution; skip tool and report error in trace |
| **Response Generation** | LLM timeout / 5xx | No response | Exponential backoff retry logic + fallback model |
| **Memory Write-back** | DB write failure | Lost history | Non-blocking retry with limited attempts |
| **Embedding Model** | Inference failure | RAG degraded | Fallback to BM25/Keyword search |

**Design Philosophy:**
* **Graceful Degradation:** The system is built to provide an answer even if non-critical subsystems (like RAG or L3 memory) are temporarily unavailable.
* **Selective Retries:** I only retry non-deterministic nodes (LLM, Write-back). Deterministic nodes (Retrieval, Intent rules) are fail-fast.

---

### 3. Scaling Strategy

#### Memory & RAG Scaling (L3 & L4)
* **Current State:** SQLite-based storage for metadata and Chroma for vectors.
* **Roadmap:**
    * **Sharding:** Horizontal partitioning of the `user_facts` and `session_turns` tables by `user_id`.
    * **Distributed Vectors:** Transitioning from local Chroma to a distributed vector database (e.g., Pinecone or Weaviate) for multi-region availability.
    * **Fact Pruning:** Implementing a decay function to archive "stale" L3 facts, keeping the active context window lean.

#### Orchestration Scaling
* **Asynchronous Model:** Moving from `ThreadPoolExecutor` to a fully `asyncio` execution model to handle higher concurrent request volumes.
* **Task Queues:** Offloading heavy `memory_write_back` and summary generation tasks to background workers (e.g., Celery/Redis) to clear the request-response cycle faster.

#### LLM Strategy
* **Model Agnosticism:** The system is already model-agnostic. Scaling involves implementing **Response Caching** (Semantic Caching) to intercept repeated queries before they hit the LLM.
* **Streaming:** Future support for Server-Sent Events (SSE) to reduce perceived latency for the end user.

---

### 4. Current Bottlenecks
To maintain transparency, the following areas are identified as the primary latency/throughput drivers:
1.  **LLM Latency:** The primary bottleneck (managed via budget control and caching).
2.  **Embedding Generation:** Sequential embedding of queries (can be optimized with local GPU inference).
3.  **SQLite Lock Contention:** Potential bottleneck during high-concurrency writes (mitigated by moving to PostgreSQL for global scale).