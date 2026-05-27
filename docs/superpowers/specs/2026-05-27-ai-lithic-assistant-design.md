# AI Lithic Assistant — Design Spec

**Date:** 2026-05-27
**Status:** Approved for implementation

## Overview

Add a conversational AI assistant to Dibble that lets users query their lithic collection in natural language. The assistant translates queries like "show me all crested blades with platform angles over 75°" into structured SQL, executes against the in-memory collection, and explains results — all locally via an embedded LLM.

## Architecture

**Engine:** `llama-cpp-python` (in-process, no daemon, PyInstaller-compatible)
**Model:** Qwen3-4B GGUF (Q4_K_M, ~2.5GB, STEM-optimized, 32K context)
**Query engine:** DuckDB Text-to-SQL via replacement scans on Pandas DataFrames
**Output enforcement:** GBNF grammar (physically prevents malformed JSON/SQL)
**Threading:** QRunnable + pyqtSignal for async token streaming
**Model management:** `from_pretrained()` from Hugging Face, cached locally

---

## 1. Assistant Engine — `_assistant.py`

### AssistantEngine class

```python
class AssistantEngine:
    """LLM-powered natural language query engine for lithic collections."""

    def __init__(self):
        self._llm: Optional[Llama] = None
        self._model_loaded: bool = False

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        """Load Qwen3-4B GGUF model. Downloads if not cached.
        Runs in background thread; progress_cb receives (stage, pct, msg)."""

    def is_loaded(self) -> bool:
        """Check if model is ready."""

    def query(self, user_text: str, collection_df: pd.DataFrame) -> AssistantResult:
        """Full query loop: prompt → SQL → execute → explain."""
```

### AssistantResult dataclass

```python
@dataclass
class AssistantResult:
    natural_language: str       # LLM's explanation of results
    sql_query: str              # The generated SQL
    row_count: int              # Number of matching artefacts
    processing_time_s: float
    error: Optional[str]        # If query failed entirely
```

### The Query Loop

```
1. Build system prompt with:
   - Exact table schema (column names + types)
   - 3 few-shot examples of natural language → SQL
   - Restriction to read-only SELECT queries
   - Domain vocabulary mapping (e.g., "crested blade" → typology column)

2. Generate SQL via llama.cpp with GBNF SQL grammar:
   - GBNF grammar constrains token selection to valid SQL syntax only
   - Output is guaranteed to parse as SQL

3. Execute SQL via DuckDB replacement scan:
   duckdb.sql(sql).df()
   - Operates directly on in-memory Pandas DataFrame
   - Zero copying, near-instant execution

4. Self-correction loop (up to 3 attempts):
   - If DuckDB raises error → append error to prompt → regenerate SQL
   - If all 3 fail → return error message to user

5. Summarize results via second LLM call:
   - Pass filtered DataFrame summary stats to LLM
   - LLM generates natural language explanation
   - Uses GBNF JSON grammar for structured output format
```

### GBNF Grammar

A Context-Free Grammar that constrains the LLM to output only valid DuckDB SQL:

```
root ::= select-statement
select-statement ::= "SELECT" select-list "FROM" table-name where-clause? limit-clause? ";"
select-list ::= "*" | column-name ("," column-name)*
where-clause ::= "WHERE" condition ("AND" condition)*
condition ::= column-name operator value
operator ::= "=" | ">" | "<" | ">=" | "<=" | "!=" | "LIKE" | "IN"
```

This is defined as a `.gbnf` file loaded at runtime.

---

## 2. Files to Create/Modify

| File | Action | Purpose |
|---|---|---|
| `lithicore/src/lithicore/_assistant.py` | Create | AssistantEngine, query loop, GBNF grammar, DuckDB integration |
| `lithicore/src/lithicore/__init__.py` | Modify | Export new symbols |
| `lithicore/pyproject.toml` | Modify | Add duckdb>=1.0 |
| `lithicore/data/grammars/sql_query.gbnf` | Create | GBNF grammar file for SQL query generation |
| `lithicope/src/lithicope/_assistant_panel.py` | Create | Chat UI widget |
| `lithicope/src/lithicope/_main_window.py` | Modify | Add Assistant tab + menu item |

**Note:** `llama-cpp-python` is not added to pyproject.toml by default — it's an optional user-installed dependency (requires C++ compiler, varies by platform). The app works without it; the Assistant tab shows a "not available" message. Users install it via `pip install llama-cpp-python` when they want the assistant.

---

## 3. Assistant Panel UI

New tab alongside Results / Annotations / Classification:

```
┌──────────────────────────────────────────┐
│ Results │ Annotations │ Classification │ Assistant │
├──────────────────────────────────────────┤
│ ┌─── AI Lithic Assistant ──────────────┐ │
│ │                                      │ │
│ │  🔵 Welcome! Ask me anything about   │ │
│ │     your lithic collection.           │ │
│ │                                      │ │
│ │  🟢 Show me all crested blades with  │ │
│ │     platform angles over 75°          │ │
│ │                                      │ │
│ │  🔵 Found 12 crested blades matching │ │
│ │     your criteria. Average length:   │ │
│ │     62.3mm, all with platform angles │ │
│ │     between 76° and 88°. Here are    │ │
│ │     the top 3...                      │ │
│ │                                      │ │
│ │  [Show SQL ▾]                        │ │
│ │  ┌──────────────────────────────┐   │ │
│ │  │ SELECT * FROM artifacts      │   │ │
│ │  │ WHERE platform_angle > 75    │   │ │
│ │  │ AND typology = 'Crested blade'│   │ │
│ │  │ LIMIT 50                     │   │ │
│ │  └──────────────────────────────┘   │ │
│ │                                      │ │
│ ├──────────────────────────────────────┤ │
│ │ [Ask anything about your collection] [→]│
│ └──────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

- User messages in green
- Assistant responses in blue
- SQL expandable via "Show SQL" toggle
- Model loading progress shown in status bar
- Results highlight in viewer when rows are clicked

---

## 4. Model Management

- **On first use:** A dialog offers to download Qwen3-4B (~2.5GB)
  - "Download Llama model to enable AI assistant? (one-time, ~2.5GB)"
  - Shows download progress bar
  - Download via `Llama.from_pretrained("Qwen/Qwen3-4B-GGUF")`
- **Storage:** `~/.dibble/models/assistant/qwen3-4b-q4_k_m.gguf`
- **Manual option:** Users can download and place the GGUF file manually
- **Fallback:** If model not found, panel shows "AI assistant requires Qwen3-4B model. [Download] [Manual install instructions]"

---

## 5. Menu Items

```
Tools
├── ...
├── Assistant
│   ├── Open Assistant     Ctrl+Shift+A
│   └── Download Model...
└── ...
```

---

## 6. Testing Strategy

| Test | Description |
|---|---|
| `test_assistant_result_dataclass` | Construction + defaults |
| `test_sql_grammar_accepts_valid` | GBNF grammar accepts "SELECT * FROM df WHERE x > 5;" |
| `test_sql_grammar_rejects_invalid` | GBNF grammar rejects "DROP TABLE df;" |
| `test_duckdb_replacement_scan` | DuckDB runs SQL on mock DataFrame |
| `test_query_loop_basic` | Full loop on trivial query |
| `test_self_correction` | SQL error → retry → success |
| `test_self_correction_exhausted` | 3 failures → error result |
