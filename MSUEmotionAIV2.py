import sys
import json
import speech_recognition as sr
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit,
    QPushButton, QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer
from transformers import pipeline
from difflib import SequenceMatcher

emotion_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

json_path = "/Users/br/Desktop/Cone Work/PHOTOSHOP PSD/WORKS/Senior Project/StudentExpressions.json"

def load_student_responses(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return [entry["response"] for entry in data]

def get_closest_emotion(user_input, json_path, threshold=0.75):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    best_match = None
    highest_ratio = 0.0

    for entry in data:
        similarity = SequenceMatcher(None, user_input.lower(), entry["response"].lower()).ratio()
        if similarity > highest_ratio:
            best_match = entry
            highest_ratio = similarity

    if best_match and highest_ratio >= threshold:
        return {"label": best_match["emotion"], "score": highest_ratio, "source": "json"}
    else:
        return None

def detect_emotion(text):
    candidate_labels = [
        "happy", "sadness", "anger", "fear", "neutral", "surprise", "disgust",
        "overwhelmed", "relieved", "nervous", "anxious", "excited", "stressed",
        "boredom", "flustered"
    ]
    result = emotion_classifier(text, candidate_labels)
    return {"label": result["labels"][0], "score": result["scores"][0]}

class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MSU EmotionAI Detector")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.input_label = QLabel("How are you feeling today?")
        self.input_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.input_label)

        self.textbox = QLineEdit()
        self.textbox.returnPressed.connect(self.detect_emotion_from_input)
        layout.addWidget(self.textbox)

        self.button = QPushButton("Detect Emotion")
        self.button.clicked.connect(self.detect_emotion_from_input)
        layout.addWidget(self.button)

        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.result_label)

        self.voice_button = QPushButton("Speak Emotion", self)
        self.voice_button.clicked.connect(self.recognize_speech)
        layout.addWidget(self.voice_button)

        self.setLayout(layout)

        self.ask_continue_label = QLabel("")
        self.ask_continue_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.ask_continue_label)

        self.yes_button = QPushButton("Yes")
        self.yes_button.clicked.connect(self.on_continue)
        self.no_button = QPushButton("No")
        self.no_button.clicked.connect(self.on_exit)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.yes_button)
        self.button_layout.addWidget(self.no_button)
        layout.addLayout(self.button_layout)

        self.yes_button.hide()
        self.no_button.hide()

        self.save_button = QPushButton("Save Session")
        self.save_button.clicked.connect(self.save_session)
        self.save_button.hide()
        layout.addWidget(self.save_button)

        self.toggle_real_time_button = QPushButton("Turn on real-time detection")
        self.toggle_real_time_button.clicked.connect(self.toggle_real_time_detection)
        layout.addWidget(self.toggle_real_time_button)

        self.setLayout(layout)

        self.realtime_enabled = False
        self.typing_timer = QTimer(self)
        self.typing_timer.setSingleShot(True)
        self.typing_timer.timeout.connect(self.detect_emotion_from_input)

        self.detected_emotion = None
        self.selected_file = None

    def recognize_speech(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        try:
            with mic as source:
                self.result_label.setText("Listening...")
                QApplication.processEvents()
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                self.textbox.setText(text)
        except sr.UnknownValueError:
            self.result_label.setText("Sorry I didn't catch that")
        except sr.RequestError:
            self.result_label.setText("Speech Recognition isn't working properly right now, try again")
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")

    def on_text_changed(self):
        self.typing_timer.start(1000)

    def detect_emotion_from_input(self):
        text = self.textbox.text().strip()
        if not text:
            return

        match = get_closest_emotion(text, json_path)
        if match:
            label = match["label"]
            confidence = match["score"]
        else:
            model_result = detect_emotion(text)
            label = model_result["label"]
            confidence = model_result["score"]

        self.result_label.setText(f"Emotion: {label} (Confidence: {confidence:.2f})")
        self.detected_emotion = label
        self.ask_continue_label.setText("Do you want to continue?")
        self.yes_button.show()
        self.no_button.show()
        self.save_button.hide()

    def toggle_real_time_detection(self):
        self.realtime_enabled = not self.realtime_enabled
        if self.realtime_enabled:
            self.textbox.textChanged.connect(self.on_text_changed)
            self.toggle_real_time_button.setText("Turn off real-time detection")
        else:
            try:
                self.textbox.textChanged.disconnect(self.on_text_changed)
            except TypeError:
                pass
            self.toggle_real_time_button.setText("Turn on real-time detection")

    def on_continue(self):
        self.ask_continue_label.setText("Click below to save your session:")
        self.save_button.show()
        self.yes_button.hide()
        self.no_button.hide()

    def save_session(self):
        if not self.selected_file:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Session", "", "Text Files (*.txt);;All Files (*)", options=options)
            if file_name:
                self.selected_file = file_name

        if self.selected_file:
            try:
                with open(self.selected_file, "a") as f:
                    session_data = {
                        "input_text": self.textbox.text(),
                        "emotion": self.result_label.text(),
                        "confidence": self.result_label.text().split('(')[1].split(')')[0]
                    }
                    f.write(json.dumps(session_data, indent=4) + "\n")
                    self.ask_continue_label.setText("Session saved!")
                    self.save_button.hide()
            except Exception as e:
                print(f"Error saving session: {e}")

    def on_exit(self):
        self.reset_ui()
        QApplication.quit()

    def reset_ui(self):
        self.textbox.setDisabled(False)
        self.textbox.clear()
        self.result_label.clear()
        self.ask_continue_label.clear()

        self.yes_button.hide()
        self.no_button.hide()
        self.save_button.hide()

        self.button.setText("Detect Emotion")
        self.button.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())
