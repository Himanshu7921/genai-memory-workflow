## Context Management: Budgeting, Priorities, and Eviction

This document details the **Context Budget System** implemented in `context_builder.py`. In production LLM systems, the context window is a finite, expensive resource. Without active management, systems suffer from "context stuffing," leading to high costs, slower response times, and the "Lost in the Middle" phenomenon where the model ignores critical information.

---

### 1. The Allocation Strategy
We treat the model's context window (e.g., 128k for `primary`) as a financial budget. Each component of the prompt is allocated a maximum "spend" in tokens.

**Default Allocations (`DEFAULT_ALLOCATION`):**
| Section | Budget (Tokens) | Description |
| :--- | :--- | :--- |
| **Response Headroom** | 8,000 | Reserved for the LLM's actual completion output. |
| **Retrieved Docs (RAG)** | 14,000 | The largest chunk, reserved for external knowledge chunks. |
| **Recent Turns** | 6,000 | Raw conversation history for immediate flow. |
| **L3 User Facts** | 2,500 | Durable user truths (protected + user facts). |
| **L2 Summary** | 2,000 | The compressed narrative of the whole session. |
| **System Prompt** | 1,200 | Core instructions and persona. |
| **Tools** | 1,000 | Recent tool outputs/results. |
| **Current Turn** | 500 | The user's active query (High Priority). |

---

### 2. The Priority Ladder (Deterministic Eviction)
When the total content exceeds the `MODEL_BUDGET`, the `ContextBuilder` does not truncate blindly. It follows a **Priority Ladder** to preserve the most critical information for an accurate response.



**Eviction Order (First to Drop → Last to Drop):**
1.  **L2 Summary Compression:** The summary is further truncated/compressed first as it is the most redundant.
2.  **Recent Conversation Turns:** Oldest turns are popped one by one. While flow is important, the "now" is more important.
3.  **RAG Chunks:** Lower-score chunks are removed. We prefer fewer, higher-relevance documents over many low-relevance ones.
4.  **L3 User Facts:** Facts are trimmed only if the budget is still exceeded. User facts are popped, then Protected facts.
5.  **Tool Outputs:** These are kept until the very end because they often contain the specific data requested in the current turn.
6.  **Current Turn:** **Never evicted.** This is the anchor of the user's intent.

---

### 3. Implementation Details

#### Token Estimation
The system uses `tiktoken` for precise counting. To ensure operational resilience, if the tokenizer package is missing, it falls back to a deterministic heuristic:
$$Tokens \approx \frac{Length(Chars)}{4}$$

#### The "Scoped State" Pattern
The `ContextBuilder` does not just return a string; it returns a `ContextBuildResult` containing a **Scoped State**. 
* This is a copy of the `OrchestrationState` modified to match what survived the eviction.
* This ensures that the LLM only "sees" what the system has audited, preventing discrepancies between what is logged and what is sent to the model.

---

### 4. Auditability & Observability
Every request generates a `budget_audit`. This allows developers to debug exactly why a model might have missed a specific fact.

**Example Audit Log:**
```json
{
  "model": "primary-128000",
  "budget_total": 128000,
  "allocation": {
    "l3_user_facts": { "tokens": 2100, "pct": 1.64 },
    "retrieved_docs": { "tokens": 14000, "pct": 10.94 },
    "unused": { "tokens": 98500, "pct": 76.95 }
  },
  "eviction_events": [
    "Reduced RAG chunks from 15 -> 10",
    "Dropped 2 oldest turns"
  ]
}
```

---

### 5. Cache Strategy Integration
By maintaining a deterministic prompt construction flow, we maximize **Prompt Caching** efficiency:
* **Static Headers:** The System Prompt and L3 Facts (which change slowly) are placed at the beginning of the prompt.
* **Volatile Tail:** The Current Turn and Tool Results are placed at the end.
* **Benefit:** This structure allows the LLM provider to cache the prefix of the prompt, significantly reducing "Time to First Token" (TTFT) and costs for multi-turn conversations.

---

### Summary of Failure Prevention
* **No "Blind Truncation":** Prevents the model from losing the user's actual question.
* **Tool Isolation:** Ensures a successful tool result is actually visible to the generator.
* **Headroom Guarantee:** Ensures the model always has enough space to actually finish its sentence without being cut off by the provider's limit.

Does this allocation balance look right for your specific use case, or do you anticipate needing a larger budget for RAG relative to conversation history?