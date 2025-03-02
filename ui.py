from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QListWidget, QLineEdit, QCompleter
import sys

class RecommenderApp(QWidget):
    def __init__(self, recommender):
        super().__init__()
        self.recommender = recommender
        self.setWindowTitle('Netflix Recommender System')
        self.setGeometry(100, 100, 400, 300)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel('Select a Movie/Show:')
        layout.addWidget(self.label)


        self.input = QLineEdit()
        completer = QCompleter(self.recommender.data_loader.titles)
        self.input.setCompleter(completer)
        self.input.textChanged.connect(self.show_recommendations)
        layout.addWidget(self.input)

        '''
        self.combo_box = QComboBox()
        self.combo_box.addItems(sorted(self.recommender.data_loader.titles))
        self.combo_box.currentIndexChanged.connect(self.show_recommendations)
        layout.addWidget(self.combo_box)
        '''

        self.recommendations_label = QLabel('Recommendations:')
        layout.addWidget(self.recommendations_label)
        self.recommendations_list = QListWidget()
        layout.addWidget(self.recommendations_list)

        self.setLayout(layout)

    def show_recommendations(self):
        selected_title = self.input.text()
        recommendations = self.recommender.get_recommendations(selected_title)
        self.recommendations_list.clear()
        self.recommendations_list.addItems(recommendations)

def launch_app(recommender):
    app = QApplication(sys.argv)
    window = RecommenderApp(recommender)
    window.show()
    sys.exit(app.exec_())
