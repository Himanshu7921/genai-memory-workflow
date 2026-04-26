# Architectural Decision Records (ADR)

This document records key architectural decisions made during the development of the system, including alternatives considered and reasoning behind final choices.

---

## ADR-001: Hybrid Memory Retrieval (Embedding + Rule-Based)

**Status:** Accepted  

### Context  
The system needs to retrieve user facts (L3 memory) reliably for both:
- Structured queries (e.g., "What is my name?")
- Semantic queries (e.g., "What do I like?")

### Options  
- **Option A:** Pure rule-based retrieval (keyword / key matching)  
- **Option B:** Pure embedding-based retrieval  
- **Option C:** Hybrid approach (rule-based + embedding similarity)

### Decision  
**Option C — Hybrid approach** was chosen.

### Rationale  
- Rule-based retrieval is fast and deterministic for structured facts  
- Embeddings enable semantic understanding for unstructured queries  
- Hybrid ensures both accuracy and flexibility  

### Consequences  
- Slightly higher complexity  
- Better real-world performance and coverage  
- Scales to both structured and unstructured memory  

---

## ADR-002: Embedding-Based Gating for Memory Injection

**Status:** Accepted  

### Context  
Injecting all memory into the LLM prompt causes:
- Token overflow  
- Irrelevant context  
- Increased latency and cost  

### Options  
- **Option A:** Inject all memory blindly  
- **Option B:** Rule-based filtering (keyword match)  
- **Option C:** Embedding-based relevance gating  

### Decision  
**Option C — Embedding-based gating** was implemented.

### Rationale  
- Enables semantic relevance filtering  
- Prevents noise in prompt  
- Improves response quality  

### Consequences  
- Requires embedding generation  
- Slight latency increase  
- Significant improvement in prompt efficiency  

---

## ADR-003: State-Machine-Based Orchestration Graph

**Status:** Accepted  

### Context  
System involves multiple steps:
- memory retrieval  
- document retrieval  
- tool execution  
- response generation  

### Options  
- **Option A:** Linear pipeline  
- **Option B:** LLM-driven agent (fully dynamic)  
- **Option C:** State-machine orchestration graph  

### Decision  
**Option C — State-machine graph** was chosen.

### Rationale  
- Enables parallel execution (memory + RAG)  
- Allows selective retries  
- Improves observability via trace events  
- Provides modular and extensible architecture  

### Consequences  
- Slightly more complex implementation  
- Stronger control and reliability  

---

## ADR-004: SQLite for Memory Storage

**Status:** Accepted  

### Context  
System requires persistent storage for:
- session memory (L2)  
- user facts (L3)  

### Options  
- **Option A:** SQLite  
- **Option B:** PostgreSQL / distributed DB  
- **Option C:** In-memory only  

### Decision  
**Option A — SQLite** was chosen.

### Rationale  
- Simple setup  
- Zero external dependencies  
- Sufficient for assignment scope  

### Consequences  
- Limited scalability at large scale  
- Easy to migrate later  

---

## ADR-005: Hybrid RAG Retrieval (Vector + Keyword + Reranking)

**Status:** Accepted  

### Context  
Document retrieval must be:
- Accurate (semantic understanding)  
- Robust (handle edge cases)  

### Options  
- **Option A:** Pure vector search  
- **Option B:** Pure keyword search  
- **Option C:** Hybrid (vector + keyword + reranking)  

### Decision  
**Option C — Hybrid retrieval** was implemented.

### Rationale  
- Vector search captures semantic meaning  
- Keyword search ensures exact matches  
- Reranking improves final relevance  

### Consequences  
- Slight increase in complexity  
- Much higher retrieval accuracy  

---

## ADR-006: Strict JSON Enforcement for LLM Outputs

**Status:** Accepted  

### Context  
LLM outputs were causing parsing failures:
- Invalid JSON  
- Extra text / formatting  

### Options  
- **Option A:** Accept free-form responses  
- **Option B:** Strict JSON enforcement + sanitization  

### Decision  
**Option B — Strict JSON enforcement** was implemented.

### Rationale  
- Required for tool planning and orchestration  
- Ensures predictable system behavior  

### Consequences  
- Requires prompt constraints  
- Added sanitization + retry logic  
- Improved system stability  

---

## ADR-007: Seed Script for System Initialization

**Status:** Accepted  

### Context  
Manual testing required repeated setup of:
- user data  
- session history  
- documents  

### Options  
- **Option A:** Manual setup  
- **Option B:** Automated seed script  

### Decision  
**Option B — Seed script** was implemented.

### Rationale  
- Enables reproducible testing  
- Reduces setup time  
- Demonstrates full system capability instantly  

### Consequences  
- Adds maintenance overhead  
- Improves developer experience significantly  

---

# Summary

These decisions collectively enable:

- Controlled orchestration  
- Efficient memory management  
- Robust retrieval  
- Scalable architecture  
- Reliable LLM interaction  