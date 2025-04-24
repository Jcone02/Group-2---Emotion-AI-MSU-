import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QStackedWidget,
    QSizePolicy,
    QScrollArea,
    QFileDialog
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QRect
from PyQt5.QtGui import QFont, QIcon, QColor, QPixmap
from transformers import pipeline
import speech_recognition as sr
from deepface import DeepFace
import cv2

class ChatBubble(QWidget):
    def __init__(self, text, is_user=True):
        super().__init__()
        layout = QHBoxLayout()
        label = QLabel(text)
        label.setWordWrap(True)
        label.setFont(QFont("Arial", 10))
        label.setStyleSheet("""
            QLabel {
                padding: 10px;
                border-radius: 10px;
                background-color: #4A90E2;
                color: white;
            }
        """ if is_user else """
            QLabel {
                padding: 10px;
                border-radius: 10px;
                background-color: #E0E0E0;
                color: black;
            }
        """)
        if is_user:
            layout.addStretch()
            layout.addWidget(label)
        else:
            layout.addWidget(label)
            layout.addStretch()
        self.setLayout(layout)

class EmotionDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion AI Detector")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("background-color: #f6f7fb;")

        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.labels = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"]

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.stacked_widget = QStackedWidget()
        self.menu_widget = self.create_menu_widget()
        self.chat_widget = self.create_chat_widget()

        self.stacked_widget.addWidget(self.menu_widget)
        self.stacked_widget.addWidget(self.chat_widget)

        layout = QVBoxLayout()
        layout.addWidget(self.stacked_widget)
        self.central_widget.setLayout(layout)

    def create_menu_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Emotion AI Detector")
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setStyleSheet("color: #333333;")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Select a mode to begin detecting emotions.")
        subtitle.setFont(QFont("Arial", 14))
        subtitle.setStyleSheet("color: #666666;")
        subtitle.setAlignment(Qt.AlignCenter)

        text_button = QPushButton("Text Input (BART)")
        voice_button = QPushButton("Voice Input")
        camera_button = QPushButton("Camera Mode")
        save_button = QPushButton("Save Session")
        exit_button = QPushButton("Exit App")

        for btn in [text_button, voice_button, camera_button, save_button, exit_button]:
            btn.setFixedHeight(50)
            btn.setFont(QFont("Arial", 12, QFont.Bold))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4A90E2;
                    color: white;
                    border: none;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #357ABD;
                }
            """)

        text_button.clicked.connect(lambda: self.switch_to_chat("text"))
        voice_button.clicked.connect(self.voice_input)
        camera_button.clicked.connect(self.camera_mode)
        save_button.clicked.connect(self.save_session)
        exit_button.clicked.connect(self.close)

        # Positioning the Exit button at the top-right
        exit_button.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                color: black;
                border: none;
                border-radius: 8px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #C4C4C4;
            }
        """)
        exit_button.setFixedSize(80, 40)
        exit_button.setGeometry(820, 10, 80, 40)

        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(30)
        layout.addWidget(text_button)
        layout.addWidget(voice_button)
        layout.addWidget(camera_button)
        layout.addWidget(save_button)
        layout.addWidget(exit_button)
        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def create_chat_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_container.setLayout(self.chat_layout)
        self.scroll_area.setWidget(self.chat_container)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message...")
        self.user_input.setFont(QFont("Arial", 11))
        self.user_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #cccccc;
                border-radius: 8px;
            }
        """)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.user_input)

        send_button = QPushButton("Send")
        send_button.setFixedHeight(40)
        send_button.setFont(QFont("Arial", 10, QFont.Bold))
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        send_button.clicked.connect(self.handle_user_input)

        input_layout.addWidget(send_button)

        # Back Button
        back_button = QPushButton("Back to Menu")
        back_button.setFixedHeight(40)
        back_button.setFont(QFont("Arial", 10, QFont.Bold))
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                color: black;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #C4C4C4;
            }
        """)
        back_button.clicked.connect(self.switch_to_menu)

        layout.addWidget(self.scroll_area)
        layout.addLayout(input_layout)
        layout.addWidget(back_button)

        widget.setLayout(layout)
        return widget

    def switch_to_chat(self, mode):
        self.chat_layout.addWidget(ChatBubble(f"Switched to {mode} mode.", is_user=False))
        self.stacked_widget.setCurrentWidget(self.chat_widget)

    def switch_to_menu(self):
        self.stacked_widget.setCurrentWidget(self.menu_widget)

    def handle_user_input(self):
        text = self.user_input.text().strip()
        if text:
            self.chat_layout.addWidget(ChatBubble(text, is_user=True))
            self.user_input.clear()
            emotion = self.detect_emotion(text)
            self.chat_layout.addWidget(ChatBubble(f"Detected emotion: {emotion}", is_user=False))
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def detect_emotion(self, text):
        result = self.classifier(text, self.labels)
        return result["labels"][0] if result["labels"] else "unknown"

    def save_session(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Session", "", "Text Files (*.txt)")
        if filename:
            with open(filename, "w") as file:
                for i in range(self.chat_layout.count()):
                    widget = self.chat_layout.itemAt(i).widget()
                    if isinstance(widget, ChatBubble):
                        # Get the text from the QLabel inside the ChatBubble
                        label = widget.findChild(QLabel)
                        file.write(f"{label.text()}\n")

    def voice_input(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.chat_layout.addWidget(ChatBubble("Listening for your voice input...", is_user=False))
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            self.chat_layout.addWidget(ChatBubble(f"You said: {text}", is_user=True))
            emotion = self.detect_emotion(text)
            self.chat_layout.addWidget(ChatBubble(f"Detected emotion: {emotion}", is_user=False))
        except sr.UnknownValueError:
            self.chat_layout.addWidget(ChatBubble("Sorry, I couldn't understand that.", is_user=False))
        except sr.RequestError:
            self.chat_layout.addWidget(ChatBubble("Could not request results from Google Speech Recognition service.", is_user=False))

    def camera_mode(self):
        self.chat_layout.addWidget(ChatBubble("Detecting emotions from your face...", is_user=False))
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']

            self.chat_layout.addWidget(ChatBubble(f"Detected emotion: {dominant_emotion}", is_user=False))
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
            cv2.imshow("Emotion Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the camera mode
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionDetectorApp()
    window.show()
    sys.exit(app.exec_())
