import sys
from PySide6.QtWidgets import QPushButton, QApplication
from PySide6.QtCore import Slot


@Slot()
def say_hello():
    print("Button clicked, Hello!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    btn = QPushButton("Click me")
    btn.clicked.connect(say_hello)
    btn.show()
    sys.exit(app.exec_())
