"""main.py — Application entry point for the Lithic Analysis Platform.

exports: main() -> None
used_by: CLI script entry point
rules:   Create QApplication, instantiate MainWindow, exec event loop.
         Must call sys.exit(app.exec()) for clean shutdown.
agent:   deepseek-v4-flash | 2026-05-26 | Initial scaffolding
"""

import sys
from PyQt6.QtWidgets import QApplication
from lithicope._main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Lithic Analysis Platform")
    app.setOrganizationName("Digital Heritage Research")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
