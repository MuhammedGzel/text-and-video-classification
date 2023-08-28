# text-and-video-classification
Deep learning models have been trained to automatically perform text and video classification tasks using deep learning techniques. Two distinct models were trained for each classification problem. For text classification, LSTM and Bidirectional LSTM models were employed, while for video classification, 3D CNN and 2D CNN+LSTM models were utilized. The TC32 dataset was employed for training the text classification models, whereas a subset of 25 classes from the UCF-101 dataset was used for training the video classification models.

Moreover, a PyQt5 interface has been developed to facilitate class prediction tasks using the trained models. This interface also enables text classification using audio and video files.

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
