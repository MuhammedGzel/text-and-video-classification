# text-and-video-classification
Deep learning models have been trained to perform text and video classification tasks automatically. For both classification problems, two distinct models have been trained. For text classification, LSTM and bidirectional LSTM models have been trained, while for video classification, 3D CNN and 2D CNN+LSTM models have been employed. The TC32 dataset has been used for training the text classification models, and for video classification training, a subset of 25 classes from the UCF-101 dataset has been utilized. 
In addition, an interface using PyQt5 has been developed for class prediction operations using the trained models. Through this interface, text classification tasks can also be conducted using audio and video files.

# Requirements
- keras==2.10.0
- keras_nightly==2.5.0.dev2021032900
- Keras_Preprocessing==1.1.2
- matplotlib==3.7.1
- moviepy==1.0.3
- nltk==3.8.1
- numpy==1.24.2
- opencv_python==4.5.2.52
- opencv_python_headless==4.5.2.52
- pandas==1.5.3
- PyQt5==5.15.9
- PyQt5_sip==12.11.1
- python_vlc==3.0.18122
- scikit_learn==1.2.2
- seaborn==0.12.2
- setuptools==68.1.2
- SpeechRecognition==3.10.0
- tensorflow==2.10.0


# Usage
- Download repo and just run "StackApplet.java", application will run.

![alt](https://github.com/MuhammedGzel/stack-data-structure-applet-visualization/blob/main/app_screenshot.png)
