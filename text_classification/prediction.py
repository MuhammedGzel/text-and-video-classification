import mimetypes
from time import sleep
import numpy as np
import moviepy.editor as mp
import os
import speech_recognition as sr
import pickle
from keras.models import load_model
from keras.utils import pad_sequences
from .clear_text_data import prepare_data


def determine_media_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type.startswith('audio'):
        return "audio"
    elif mime_type.startswith('video'):
        return "video"
    else:
        return "unknown"


def convert_sound_to_text(file_path):
    try:
        with mp.AudioFileClip(file_path) as audio_clip:
            audio_duration = audio_clip.duration

            if audio_duration > 60:
                return False, ("Sound length longer than 1 minute will not be converted. You can use a maximum of 1 "
                               "minute of sound for text classification from sound.")

            recognizer = sr.Recognizer()
            with sr.AudioFile(file_path) as source:
                audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language="tr-TR")
                return True, text
            except sr.UnknownValueError:
                return False, "Failed to detect sound"
            except sr.RequestError as e:
                return False, f"Error: {e}"

    except Exception as e:
        return False, e


def convert_video_to_text(file_path):
    try:
        with mp.VideoFileClip(file_path) as video_clip:
            if video_clip.duration > 60:
                return False, ("Video length longer than 1 minute will not be converted. You can use a maximum of 1 "
                               "minute of video for text classification from video.")

            audio_file_path = "temp_audio.wav"
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_file_path)
            converted_text = convert_sound_to_text(audio_file_path)
            os.remove(audio_file_path)
            return converted_text

    except Exception as e:
        return False, e


def convert_media_to_text(file_path):
    try:
        media_type = determine_media_type(file_path)
        if media_type in "audio":
            return convert_sound_to_text(file_path)
        elif media_type in "video":
            return convert_video_to_text(file_path)
    except Exception as e:
        print(e)


def predict_text_class(text, model_name):
    try:
        classes = list(map(lambda x: x.strip(), open('text_classification/networks/label_dict.txt')))
        model_path = "text_classification/networks/" + model_name + "/text_net_" + model_name + ".h5"
        tokenizer_path = "text_classification/networks/" + model_name + "/tokenizer_" + model_name + ".pickle"
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        clean_text = prepare_data(text)
        sequences = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(sequences, maxlen=164)
        predicted = model.predict(padded)
        index = np.argmax(predicted, axis=1)
        predicted_class = classes[index[0]]
        return True, predicted_class
    except Exception as e:
        print(e)
        return False, "An error occurred during the text classification process."


