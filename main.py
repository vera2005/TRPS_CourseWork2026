"""Точка входа приложения «Турник-Ассистент»."""

import sys
import traceback

from PyQt6.QtWidgets import QApplication, QMessageBox

from ui_main import MainWindow


def handle_unexpected_error(exc_type, exc_value, exc_traceback) -> None:
    """Показывает неперехваченную ошибку вместо молчаливого завершения."""
    details = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )
    print(details, file=sys.stderr)
    QMessageBox.critical(
        None,
        "Критическая ошибка",
        f"Приложение столкнулось с непредвиденной ошибкой:\n{exc_value}",
    )


def main() -> int:
    """Создаёт QApplication, открывает главное окно и запускает цикл событий."""
    sys.excepthook = handle_unexpected_error
    app = QApplication(sys.argv)
    app.setApplicationName("Турник-Ассистент")
    app.setOrganizationName("BMSTU_IU6")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())