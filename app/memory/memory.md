## Memory Architecture: The L1–L4 Hierarchy

This document outlines the multi-tiered memory strategy used in this system. Most LLM applications suffer from "context amnesia" or "token bloat" because they treat all history as a single, flat string. This architecture solves that by categorizing data based on its **volatility, durability, and structure.**

This architecture separates conversational state (L2), persistent identity (L3), and external knowledge (L4), enabling controlled context injection instead of naive history concatenation.

---

### Layer 1: Working Memory (Transient)
**Location:** `OrchestrationState` & `WorkingMemoryTurn`  
**Lifetime:** Single Request/Response Cycle

L1 memory is the "scratchpad" for the orchestration graph. It exists only while the request is being processed.

* **Contents:** The current user message, intermediate tool outputs, retrieved document chunks, and the calculated intent.
* **Purpose:** To provide a shared state for all nodes in the graph (e.g., the `tool_execution` node writes to L1 so the `generate` node can read from it).
* **Cleanup:** Once the response is sent and the write-back is complete, L1 is purged from memory.

---

### Layer 2: Session Memory (Short-Term Persistence)
**Location:** SQLite `sessions` and `session_turns` tables  
**Lifetime:** Active Conversation Session

L2 memory provides continuity across multiple turns within a single session.

* **Session Summary:** A compressed string representation of the conversation so far. This is updated via the `write_back` node after every turn to avoid repeating the entire history.
* **Protected Facts:** A lossless layer for high-value data (dates, specific amounts, identifiers) that must not be "fuzzy" or lost during summarization.
* **Recent Turns:** The last $N$ messages are kept in raw format to maintain conversational flow and local reference (e.g., "What did I just say?").

---

### Layer 3: User Facts (Long-Term Structured Memory)
**Location:** SQLite `user_facts` table  
**Lifetime:** Permanent (across all sessions for a specific `user_id`)

L3 memory is what makes the assistant "know" the user over months or years. Unlike L2, which is conversational, L3 is **declarative**.

* **Structure:** Data is stored as key-value pairs with metadata (e.g., `key: "dietary_restriction", value: "vegan", confidence: 0.9`).
* **Conflict Resolution:** When a new fact is extracted that contradicts an old one (e.g., a user changed their `risk_tolerance`), the system uses a "Latest-Wins" policy. The old fact is marked as `superseded_fact_id` rather than deleted, preserving an audit trail.
* **Extraction:** Primarily deterministic or high-confidence LLM extraction during the `write_back` phase.
* **Semantic Retrieval (Extension):** In addition to deterministic key-based lookup, the system supports embedding-based similarity search for unstructured user preferences (e.g., "I enjoy football" → "what do I like?"). Embeddings are generated at write-time and stored alongside facts to enable efficient semantic retrieval.

---

### Layer 4: RAG Documents (Global Knowledge)
**Location:** SQLite (`corpus_documents` metadata) + Vector Store (ChromaDB for embeddings)
**Lifetime:** Indefinite (Static or Administrative update)

L4 is the "library" of external knowledge that the model was not natively trained on or that requires strict grounding.

* **Indexing:** Documents are split into chunks via `chunking.py` and embedded using the `all-MiniLM-L6-v2` model.
* **Hybrid Retrieval:** Uses a combination of vector similarity (semantic) and BM25-style keyword matching (lexical) to ensure documents are found even if the user uses slightly different terminology.
* **Isolation:** L4 data is strictly read-only for the user; the model can cite it but cannot change it.


---

### Memory Pipeline: The Lifecycle of a Fact

1.  **Extraction:** During the `write_back` node, the system scans the turn for L3 facts using `user.py`.
2.  **Conflict Check:** The `policies.py` engine checks if the user is updating an existing fact.
3.  **Persistence:** Data is written to `memory.db`.
4.  **Hydration:** On the next request, `memory_retrieve.py` runs a parallel query to pull relevant L2 summaries and L3 facts based on the user ID.
5.  **Gating:** The `generate.py` node decides if the retrieved memory is actually relevant to the current query before injecting it into the prompt.

---

### Context Integration Note

Memory from all layers is not blindly injected into the LLM. The `generate.py` node applies a **context budget and relevance gating mechanism**, ensuring only the most relevant L2, L3, and L4 data is included in the final prompt. This prevents token overflow, reduces latency, and improves response quality.

### Summary Table: Layer Comparison

| Layer | Scope | Format | Storage | Retrieval Method |
| :--- | :--- | :--- | :--- | :--- |
| **L1** | Request | Object | RAM | Direct Access |
| **L2** | Session | Summary/Text | SQLite | `session_id` Lookup |
| **L3** | User | Structured (KV) | SQLite | `user_id` + Key Match |
| **L4** | Global | Vector/Chunks | Chroma DB | Hybrid Search |