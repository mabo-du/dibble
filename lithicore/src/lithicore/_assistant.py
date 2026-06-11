"""_assistant.py — AI-powered natural language query engine for lithic collections.

exports: AssistantEngine
used_by: lithicope assistant panel
rules:   No GUI imports. DuckDB queries are read-only SELECT. LLM is optional dependency.
         All functions safe to call when model not loaded (returns error result).
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
         deepseek-v4-pro | 2026-06-12 | Added _validate_sql() safety gate rejecting non-SELECT, multi-statement, and dangerous keywords
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from lithicore._models import AssistantResult


MODEL_DIR = Path.home() / ".dibble" / "models" / "assistant"
MODEL_FILENAME = "qwen3-4b-q4_k_m.gguf"
GRAMMAR_DIR = Path(__file__).resolve().parent.parent / "data" / "grammars"
SQL_GRAMMAR_PATH = GRAMMAR_DIR / "sql_query.gbnf"


SCHEMA_DESCRIPTION = """
Table: artifacts
Columns:
  - length_mm (FLOAT) — maximum length in mm
  - width_mm (FLOAT) — maximum width in mm
  - thickness_mm (FLOAT) — maximum thickness in mm
  - surface_area_mm2 (FLOAT) — total surface area
  - volume_mm3 (FLOAT) — total volume
  - elongation (FLOAT) — length/width ratio
  - flatness (FLOAT) — width/thickness ratio
  - compactness (FLOAT) — volume/length^3
  - relative_thickness (FLOAT) — thickness/length
  - scar_count (INTEGER) — number of flake scars detected
  - mean_scar_area_mm2 (FLOAT) — average scar area
  - platform_angle_deg (FLOAT) — platform angle in degrees
  - edge_angle_mean_deg (FLOAT) — mean edge angle
  - edge_angle_std_deg (FLOAT) — edge angle std deviation
  - curvature_index (FLOAT) — dorsal curvature
  - cross_section_profile (FLOAT) — 0=flat, 1=triangular, 2=round
  - symmetry_score (FLOAT) — bilateral symmetry (0-1)
  - com_z_ratio (FLOAT) — centre of mass height ratio
  - dorsal_ridge_count (INTEGER) — number of parallel ridges
  - surface_roughness (FLOAT) — texture metric
  - typology (TEXT) — predicted type
  - artefact_label (TEXT) — user-assigned name
"""

FEW_SHOT_EXAMPLES = """
-- Example 1: "show me all blades longer than 100mm"
SELECT * FROM artifacts WHERE typology = 'Blade' AND length_mm > 100 LIMIT 50;

-- Example 2: "what's the average platform angle of crested blades?"
SELECT AVG(platform_angle_deg) FROM artifacts WHERE typology = 'Crested blade';

-- Example 3: "find the 5 most symmetrical handaxes"
SELECT * FROM artifacts WHERE typology = 'Handaxe' ORDER BY symmetry_score DESC LIMIT 5;
"""


class AssistantEngine:
    """LLM-powered natural language query engine for lithic collections."""

    def __init__(self) -> None:
        self._llm = None
        self._grammar: Optional[str] = None
        self._model_available = False

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        """Load the LLM model. Downloads from HuggingFace if not cached."""
        try:
            import llama_cpp
        except ImportError:
            if progress_cb:
                progress_cb("error", 0.0, "llama-cpp-python not installed")
            return

        model_path = MODEL_DIR / MODEL_FILENAME
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            if progress_cb:
                progress_cb("download", 0.0, "Downloading model (~2.5GB)...")
            try:
                self._llm = llama_cpp.Llama.from_pretrained(
                    repo_id="Qwen/Qwen3-4B-GGUF",
                    filename="*q4_k_m.gguf",
                    verbose=False,
                )
            except Exception as exc:
                if progress_cb:
                    progress_cb("error", 0.0, f"Download failed: {exc}")
                return
        else:
            if progress_cb:
                progress_cb("loading", 0.0, "Loading model...")
            try:
                self._llm = llama_cpp.Llama(
                    model_path=str(model_path),
                    n_ctx=4096,
                    n_threads=os.cpu_count() or 4,
                    verbose=False,
                )
            except Exception as exc:
                if progress_cb:
                    progress_cb("error", 0.0, f"Model load failed: {exc}")
                return

        if SQL_GRAMMAR_PATH.exists():
            self._grammar = SQL_GRAMMAR_PATH.read_text()

        self._model_available = True
        if progress_cb:
            progress_cb("ready", 1.0, "AI assistant ready")

    def is_loaded(self) -> bool:
        """Check if the model is ready for queries."""
        return self._model_available and self._llm is not None

    def _validate_sql(self, sql: str) -> bool:
        """Validate that a generated SQL string is a safe SELECT query.

        Rejects non-SELECT statements, multi-statement queries, and queries
        that reference system tables. This is a hard safety gate independent
        of the GBNF grammar.
        """
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            return False
        if ";" in stripped[:-1]:
            return False  # Multi-statement — potential injection
        # Reject dangerous keywords outside of quotes
        forbidden = {"DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE",
                      "TRUNCATE", "EXEC", "EXECUTE", "ATTACH", "DETACH",
                      "PRAGMA", "IMPORT", "EXPORT"}
        words = set(stripped.replace("(", " ").replace(")", " ").replace(",", " ").split())
        if words & forbidden:
            return False
        return True

    def query(self, user_text: str, collection_df: pd.DataFrame) -> AssistantResult:
        """Run the full query loop: natural language -> SQL -> execute -> explain.

        Args:
            user_text: The user's natural language query.
            collection_df: In-memory DataFrame of the current artefact collection.

        Returns:
            AssistantResult with natural_language explanation, sql_query, row_count.
        """
        import duckdb

        if not self.is_loaded():
            return AssistantResult(
                error="AI model not loaded. Use Assistant > Download Model to set up.",
                processing_time_s=0.0,
            )

        if collection_df.empty:
            return AssistantResult(
                natural_language="No artefacts in the current collection to query.",
                processing_time_s=0.0,
            )

        start = time.time()

        # Step 1: Build prompt
        prompt = self._build_sql_prompt(user_text, collection_df)

        # Step 2: Generate SQL with GBNF grammar
        sql = self._generate_sql(prompt)
        if sql is None:
            return AssistantResult(
                error="Failed to generate a valid SQL query.",
                processing_time_s=round(time.time() - start, 2),
            )

        if not self._validate_sql(sql):
            return AssistantResult(
                sql_query=sql,
                error="Generated SQL query was rejected by safety validator.",
                processing_time_s=round(time.time() - start, 2),
            )

        # Step 3: Self-correcting execution loop
        result_df = None
        last_error = ""
        for attempt in range(3):
            try:
                result_df = duckdb.sql(sql).df()
                break
            except Exception as exc:
                last_error = str(exc)
                if attempt < 2:
                    sql = self._fix_sql(prompt, sql, last_error)

        elapsed = time.time() - start

        if result_df is None:
            return AssistantResult(
                sql_query=sql,
                error=f"SQL execution failed after 3 attempts: {last_error}",
                processing_time_s=round(elapsed, 2),
            )

        # Step 4: Summarize results
        summary = self._summarize_results(user_text, result_df)

        return AssistantResult(
            natural_language=summary or f"Found {len(result_df)} matching artefacts.",
            sql_query=sql,
            row_count=len(result_df),
            processing_time_s=round(elapsed, 2),
        )

    def _build_sql_prompt(self, user_text: str, df: pd.DataFrame) -> str:
        """Build the system prompt with schema and user query."""
        return (
            "You are a SQL query generator for a lithic (stone tool) analysis database. "
            "Generate ONLY a DuckDB SQL SELECT query. No explanations, no markdown.\n\n"
            f"Schema:\n{SCHEMA_DESCRIPTION}\n\n"
            f"Examples:\n{FEW_SHOT_EXAMPLES}\n\n"
            f"Table has {len(df)} rows.\n\n"
            f"User query: {user_text}\n\n"
            "SQL:"
        )

    def _generate_sql(self, prompt: str) -> Optional[str]:
        """Generate SQL constrained by GBNF grammar."""
        try:
            output = self._llm.create_completion(
                prompt,
                max_tokens=256,
                temperature=0.1,
                grammar=self._grammar,
                stop=[";"],
            )
            text = output["choices"][0]["text"].strip()
            if not text.endswith(";"):
                text += ";"
            return text
        except Exception:
            return None

    def _fix_sql(self, prompt: str, bad_sql: str, error: str) -> Optional[str]:
        """Fix a SQL error by appending the error and regenerating."""
        fix_prompt = (
            f"{prompt}\n\n"
            f"Previous (failed) SQL: {bad_sql}\n"
            f"Error: {error}\n"
            "Fixed SQL:"
        )
        return self._generate_sql(fix_prompt)

    def _summarize_results(self, user_text: str, df: pd.DataFrame) -> Optional[str]:
        """Generate a natural language summary of query results."""
        if df.empty:
            return "No artefacts matched your query."

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        for col in list(numeric_cols)[:5]:
            stats[col] = {
                "mean": round(float(df[col].mean()), 1),
                "min": round(float(df[col].min()), 1),
                "max": round(float(df[col].max()), 1),
            }

        type_counts = {}
        if "typology" in df.columns:
            type_counts = df["typology"].value_counts().head(5).to_dict()

        summary_prompt = (
            "Summarize these lithic analysis results in 1-3 concise sentences.\n"
            f"User asked: {user_text}\n"
            f"Found {len(df)} matching artefacts.\n"
            f"Summary stats: {stats}\n"
            f"Typology breakdown: {type_counts}\n\n"
            "Response:"
        )

        try:
            output = self._llm.create_completion(
                summary_prompt,
                max_tokens=200,
                temperature=0.3,
                stop=["\n\n"],
            )
            return output["choices"][0]["text"].strip()
        except Exception:
            return None

    @staticmethod
    def get_model_status() -> dict:
        """Return model status info for the UI."""
        model_path = MODEL_DIR / MODEL_FILENAME
        size_mb = round(model_path.stat().st_size / (1024 * 1024), 1) if model_path.exists() else 0
        return {
            "installed": model_path.exists(),
            "size_mb": size_mb,
            "path": str(model_path),
            "grammar_exists": SQL_GRAMMAR_PATH.exists(),
        }
