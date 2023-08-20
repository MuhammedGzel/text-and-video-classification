import sys
from datetime import timedelta
from multiprocessing import Process, Queue
from time import sleep
import vlc
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from text_classification.prediction import convert_media_to_text, predict_text_class
from video_classification.prediction import predict_video_class
from PyQt5.uic import loadUi


def show_message(msg_type, title, text):
    msg_box = QMessageBox()
    if msg_type == "information":
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
    if msg_type == "error":
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setStandardButtons(QMessageBox.Ok)
    if msg_type == "question":
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
    # msg_box.setWindowIcon(QIcon("GUI/images/firat_logo.png"))
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    return msg_box.exec_()


def media_to_text(file_path, process_result_queue):
    result = convert_media_to_text(file_path)
    process_result_queue.put(result)
    print(result)


def classify_text(text, model_name, process_result_queue):
    result = predict_text_class(text, model_name)
    process_result_queue.put(result)


def classify_video(video_path, model_name, process_result_queue):
    result = predict_video_class(video_path, model_name)
    process_result_queue.put(result)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("GUI/interface.ui", self)
        self.selected_model_name = None
        self.process = Process()
        self.process_result_queue = Queue()
        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        self.control_media_volume_slider.setValue(self.media_player.audio_get_volume())
        self.update_media_player_timer = QTimer(self)
        self.media_to_text_result = None
        self.write_media_text_result_timer = QTimer(self)
        self.write_classification_result_timer = QTimer(self)
        self.set_ui_initial_state()
        self.connect_signals()

    def set_ui_initial_state(self):
        self.set_select_model_widget_state(False)
        self.set_media_widget_state(False)
        self.set_plain_text_area_state(False)
        self.set_classification_widget_state(False)

    def connect_signals(self):
        self.video_classification_from_video_file_button.toggled.connect(self.set_ui_for_video_classification)
        self.text_classification_from_text_button.toggled.connect(self.set_ui_for_text_classification_from_text)
        self.text_classification_from_media_file_button.toggled.connect(self.set_ui_for_text_classification_from_media)
        self.model1_button.toggled.connect(self.set_selected_model_name)
        self.model2_button.toggled.connect(self.set_selected_model_name)
        self.media_playing_slider.sliderMoved.connect(self.set_media_position)
        self.media_playing_slider.sliderPressed.connect(self.set_media_position)
        self.play_pause_media_button.clicked.connect(self.play_pause_media)
        self.rewind_media_10s_button.clicked.connect(self.rewind_media_10_seconds)
        self.forward_media_10s_button.clicked.connect(self.forward_media_10_seconds)
        self.stop_media_button.clicked.connect(self.stop_media)
        self.replay_media_button.clicked.connect(self.replay_media)
        self.control_media_volume_slider.valueChanged.connect(self.set_media_volume)
        self.mute_unmute_media_button.clicked.connect(self.mute_unmute_media)
        self.update_media_player_timer.timeout.connect(self.update_media_player)
        self.write_media_text_result_timer.timeout.connect(self.write_media_text_result)
        self.show_hide_media_text_checkbox.stateChanged.connect(lambda state: self.media_text_area.setVisible(state))
        self.select_media_button.clicked.connect(self.open_media)
        self.classify_button.clicked.connect(self.classify)
        self.write_classification_result_timer.timeout.connect(self.write_classification_result)

    def set_selected_model_name(self):
        if self.video_classification_from_video_file_button.isChecked():
            if self.model1_button.isChecked():
                self.selected_model_name = "3d_cnn"
            else:
                self.selected_model_name = "2d_cnn_plus_lstm"
        else:
            if self.model1_button.isChecked():
                self.selected_model_name = "lstm"
            else:
                self.selected_model_name = "bidirectional_lstm"

    def set_select_model_widget_state(self, state):
        self.select_model_widget.setEnabled(state)
        self.select_model_widget.setVisible(state)

    def set_media_widget_state(self, state):
        self.media_widget.setEnabled(state)
        self.media_widget.setVisible(state)

    def set_media_text_area_state(self, state):
        self.media_text_area.setEnabled(state)
        self.media_text_area.setVisible(state)

    def set_show_hide_media_text_checkbox_state(self, state):
        self.show_hide_media_text_checkbox.setEnabled(state)
        self.show_hide_media_text_checkbox.setVisible(state)

    def set_plain_text_area_state(self, state):
        self.plain_text_area.setEnabled(state)
        self.plain_text_area.setVisible(state)

    def set_classification_widget_state(self, state):
        self.classification_widget.setEnabled(state)
        self.classification_widget.setVisible(state)

    def set_ui_for_classification(self, classification_type, model1_text, model2_text, text_widget_state):
        self.classification_type.setText(classification_type)
        self.set_select_model_widget_state(True)
        self.model1_button.setText(model1_text)
        self.model1_button.setChecked(True)
        self.model2_button.setText(model2_text)
        self.model2_button.setChecked(False)
        self.set_selected_model_name()
        self.set_media_widget_state(not text_widget_state)
        self.set_media_text_area_state(False)
        self.set_show_hide_media_text_checkbox_state(False)
        self.set_plain_text_area_state(text_widget_state)
        self.plain_text_area.setPlainText("")
        self.set_classification_widget_state(True)
        self.predicted_class_label.setText("")
        self.stop_all_operations()

    def set_ui_for_video_classification(self):
        self.set_ui_for_classification("Video Classification", "3D CNN",
                                       "2D CNN+LSTM", False)

    def set_ui_for_text_classification_from_text(self):
        self.set_ui_for_classification("Text Classification-From Text", "LSTM",
                                       "Bidirectional LSTM", True)

    def set_ui_for_text_classification_from_media(self):
        self.set_ui_for_classification("Text Classification-From Media File", "LSTM",
                                       "Bidirectional LSTM", False)

    def start_process(self, target_func, args):
        self.process_result_queue = Queue()
        self.process = Process(target=target_func, args=args + (self.process_result_queue,))
        self.process.start()

    def stop_process(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join()

    def open_media(self):
        file_path, _ = QFileDialog.getOpenFileName()
        if file_path:
            media = self.instance.media_new(file_path)
            self.media_player.set_media(media)
            self.media_player.set_hwnd(self.media_screen_frame.winId())
            self.set_media_volume()
            self.play_pause_media()
            self.play_pause_media_button.setIcon(QIcon("GUI/images/pause_media.png"))
            self.media_path_label.setText(file_path)

            if self.text_classification_from_media_file_button.isChecked():
                try:
                    self.stop_process()
                    self.start_process(media_to_text, (self.media_path_label.text(),))
                    self.set_media_text_area_state(True)
                    self.set_show_hide_media_text_checkbox_state(True)
                    self.media_text_area.setPlainText("Please wait until the selected media is converted to text.")
                    self.write_media_text_result_timer.start(1000)
                except Exception as e:
                    print(e)

    def write_media_text_result(self):
        print("write_media_text_result çalışıyor.")
        if not self.process_result_queue.empty():
            self.media_to_text_result = self.process_result_queue.get()
            self.media_text_area.setPlainText(str(self.media_to_text_result[1]))
        if self.process is None or not self.process.is_alive():
            self.write_media_text_result_timer.stop()

    def play_pause_media(self):
        if self.media_player.is_playing():
            self.media_player.pause()
            self.update_media_player_timer.stop()
            self.play_pause_media_button.setIcon(QIcon("GUI/images/play_media.png"))
        else:
            self.media_player.play()
            self.update_media_player_timer.start(100)
            self.play_pause_media_button.setIcon(QIcon("GUI/images/pause_media.png"))

    def rewind_media_10_seconds(self):
        if self.media_player.is_playing():
            new_time = max(0, self.media_player.get_time() - 10000)
            self.media_player.set_time(new_time)

    def forward_media_10_seconds(self):
        if self.media_player.is_playing():
            media_length = self.media_player.get_length()
            print(media_length)
            self.media_player.set_time(min(media_length, self.media_player.get_time() + 10000))

    def stop_media(self):
        self.media_player.stop()
        self.update_media_player_timer.stop()
        self.play_pause_media_button.setIcon(QIcon("GUI/images/play_media.png"))
        self.media_elapsed_time_label.setText("0:00:00")
        self.media_playing_slider.setValue(0)

    def replay_media(self):
        self.stop_media()
        self.play_pause_media()

    def set_media_volume(self):
        self.media_player.audio_set_volume(self.control_media_volume_slider.value())
        self.volume_level_label.setText(str(self.control_media_volume_slider.value()) + "%")

    def mute_unmute_media(self):
        if self.media_player.audio_get_mute():
            self.media_player.audio_set_mute(False)
            self.mute_unmute_media_button.setIcon(QIcon("GUI/images/unmute_media.png"))
        else:
            self.media_player.audio_set_mute(True)
            self.mute_unmute_media_button.setIcon(QIcon("GUI/images/mute_media.png"))

    def update_media_player(self):
        total_time = self.media_player.get_length()
        elapsed_time = self.media_player.get_time()
        self.media_playing_slider.setValue(int(self.media_player.get_position() * 1000))
        self.media_elapsed_time_label.setText(str(timedelta(seconds=round(elapsed_time / 1000))))
        self.media_total_time_label.setText(str(timedelta(seconds=round(total_time / 1000))))

        if not self.media_player.is_playing():
            self.update_media_player_timer.stop()
            self.play_pause_media_button.setIcon(QIcon("GUI/images/play_media.png"))

    def set_media_position(self):
        self.update_media_player_timer.stop()
        self.media_player.set_position(self.media_playing_slider.value() / 1000.0)
        self.update_media_player_timer.start(100)

    def classify(self):
        print(self.selected_model_name)
        try:
            if self.process.is_alive():
                if self.write_media_text_result_timer.isActive():
                    show_message("error", "Converting Media to Text: In Progress", "Media to Text "
                                                                                   "conversion is in progress, please "
                                                                                   "wait until the process is "
                                                                                   "complete.")
                else:
                    self.select_classification_type_frame.setEnabled(True)
                    self.classify_button.setText("Classify")
                    self.predicted_class_label.setText("")
                    self.stop_process()

            else:
                target = None
                args = None
                if self.video_classification_from_video_file_button.isChecked():
                    if self.media_path_label.text():
                        target = classify_video
                        args = (self.media_path_label.text(), self.selected_model_name,)
                    else:
                        show_message("error", "Media Not Selected", "Please select a video file.")

                elif self.text_classification_from_text_button.isChecked():
                    if self.plain_text_area.toPlainText():
                        target = classify_text
                        args = (self.plain_text_area.toPlainText(), self.selected_model_name,)
                    else:
                        show_message("error", "Empty Text Area", "Please enter a text.")

                else:
                    if self.media_to_text_result and self.media_to_text_result[0]:
                        target = classify_text
                        args = (self.media_to_text_result[1], self.selected_model_name,)
                    else:
                        show_message("error", "Classification Error", "Please select a media less "
                                                                      "than 1 minute and wait for it to be converted "
                                                                      "to text, then try again.")

                if target is not None and args is not None:
                    self.stop_process()
                    self.start_process(target_func=target, args=args)
                    self.write_classification_result_timer.start(1000)
                    self.classify_button.setText("Stop")
                    self.predicted_class_label.setText("Classification in progress, please wait.")
                    self.select_classification_type_frame.setEnabled(False)

        except Exception as e:
            print(e)

    def write_classification_result(self):
        print("write_classification_result çalışıyor.")
        if not self.process_result_queue.empty():
            result = self.process_result_queue.get()
            print(result)
            self.predicted_class_label.setText("Predicted class:" + result[1])
            self.classify_button.setText("Classify")
            self.select_classification_type_frame.setEnabled(True)
        if self.process is None or not self.process.is_alive():
            self.write_classification_result_timer.stop()

    def stop_all_operations(self):
        self.stop_process()
        self.stop_media()
        self.media_total_time_label.setText("0.00.00")

    def closeEvent(self, event):
        ques = show_message("question", "Close Program", "Are you sure you want to close the program?")
        if ques == QMessageBox.Yes:
            self.stop_all_operations()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
