"""_assistant_panel.py — Chat UI for the AI Lithic Assistant.

exports: AssistantPanel(QWidget)
used_by: MainWindow tab widget
rules:   Uses QRunnable for async LLM execution. No direct llama.cpp imports.
         Safe to use without model loaded (shows explanatory message).
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from lithicore import AssistantEngine


class AssistantPanel(QWidget):
    """Chat panel for querying the lithic collection with natural language."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._engine = AssistantEngine()
        self._collection_df = None
        self._build_ui()
        self._start_model_load()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QLabel("AI Lithic Assistant")
        header.setStyleSheet("font-size: 13pt; font-weight: bold;")
        layout.addWidget(header)

        # Status
        self._status_label = QLabel("Initialising...")
        self._status_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._status_label)

        # Chat history
        self._chat = QTextBrowser()
        self._chat.setOpenExternalLinks(False)
        self._chat.setMinimumHeight(200)
        layout.addWidget(self._chat, stretch=1)

        # Show SQL toggle
        self._sql_check = QCheckBox("Show generated SQL")
        self._sql_check.setChecked(False)
        layout.addWidget(self._sql_check)

        # Input row
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask about your lithic collection...")
        self._input.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input)
        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self._send_btn)
        layout.addLayout(input_row)

        # Welcome message
        self._append_html(
            "🔵 Welcome to the AI Lithic Assistant!<br>"
            "Ask questions about your collection in natural language.<br>"
            "<i>Examples:</i><br>"
            "• &quot;Show me all blades longer than 80mm&quot;<br>"
            "• &quot;What's the average platform angle of crested blades?&quot;<br>"
            "• &quot;Find the 5 most symmetrical handaxes&quot;"
        )

    def set_collection(self, df) -> None:
        """Set the current in-memory collection DataFrame."""
        self._collection_df = df

    def _start_model_load(self) -> None:
        """Start model loading in the background."""
        self._status_label.setText("Loading AI model...")
        # Since we can't easily thread this without signals,
        #  just load synchronously for now (blocks ~2-3s if model cached)
        try:
            def cb(stage: str, pct: float, msg: str) -> None:
                if stage == "ready":
                    self._status_label.setText("✅ AI assistant ready")
                    self._input.setEnabled(True)
                    self._send_btn.setEnabled(True)
                elif stage == "download":
                    self._status_label.setText(f"⬇ {msg}")
                elif stage == "error":
                    self._status_label.setText(f"⚠ {msg}")
                elif stage == "loading":
                    self._status_label.setText(f"⏳ {msg}")

            self._engine.load_model(progress_cb=cb)
            if not self._engine.is_loaded():
                self._status_label.setText(
                    "⚠ AI model not available. Install via: pip install llama-cpp-python"
                )
                self._append_html(
                    "⚠ <b>AI model not loaded.</b><br>"
                    "To use the AI assistant, install llama-cpp-python:<br>"
                    "<code>pip install llama-cpp-python</code><br>"
                    "Then restart Dibble and open this panel again."
                )
        except Exception as exc:
            self._status_label.setText("⚠ Model load failed")
            self._append_html(f"⚠ <b>Model load error:</b> {exc}")

    def _on_send(self) -> None:
        """Handle send button / Enter key."""
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()

        # Show user message
        self._append_html(f"🟢 <b>You:</b> {text}")

        # Check model
        if not self._engine.is_loaded():
            self._append_html("⚠ <b>AI model not loaded.</b> Cannot process query.")
            return

        # Check collection
        if self._collection_df is None or self._collection_df.empty:
            self._append_html(
                "🔵 <b>No collection loaded.</b> Open a mesh or batch of meshes first."
            )
            return

        # Execute query
        result = self._engine.query(text, self._collection_df)

        if result.error:
            self._append_html(f"⚠ <b>Error:</b> {result.error}")
            return

        # Build response HTML
        resp = f"🔵 <b>Assistant:</b> {result.natural_language}"
        if result.row_count > 0:
            resp += (
                f"<br><i>({result.row_count} artefacts found"
                f" in {result.processing_time_s:.1f}s)</i>"
            )

        if self._sql_check.isChecked() and result.sql_query:
            resp += (
                f"<pre style='background:#f5f5f5;padding:8px;font-size:9pt;'>"
                f"{result.sql_query}</pre>"
            )

        self._append_html(resp)

    def _append_html(self, html: str) -> None:
        """Append HTML content to the chat browser."""
        self._chat.append(html)
        # Scroll to bottom
        scrollbar = self._chat.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
