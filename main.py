import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, \
    QStackedWidget


class Welcome(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()

        self.title = 'Hoşgeldiniz'
        self.initUI(stacked_widget)

    def initUI(self, stacked_widget):
        layout = QVBoxLayout()
        label = QLabel('Hoşgeldiniz!')
        layout.addWidget(label)

        button1 = QPushButton('Sayfa 1')
        button1.clicked.connect(lambda: stacked_widget.setCurrentIndex(1))
        layout.addWidget(button1)

        button2 = QPushButton('Sayfa 2')
        button2.clicked.connect(lambda: stacked_widget.setCurrentIndex(2))
        layout.addWidget(button2)

        self.setLayout(layout)


class Page1(QWidget):
    def __init__(self):
        super().__init__()

        self.title = 'Sayfa 1'
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        label = QLabel('Bu sayfa 1')
        layout.addWidget(label)

        self.setLayout(layout)


class Page2(QWidget):
    def __init__(self):
        super().__init__()

        self.title = 'Sayfa 2'
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        label = QLabel('Bu sayfa 2')
        layout.addWidget(label)

        self.setLayout(layout)


app = QApplication(sys.argv)

stacked_widget = QStackedWidget()

welcome = Welcome(stacked_widget)
stacked_widget.addWidget(welcome)

page1 = Page1()
stacked_widget.addWidget(page1)

page2 = Page2()
stacked_widget.addWidget(page2)

stacked_widget.show()

sys.exit(app.exec_())