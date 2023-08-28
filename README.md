# Introduction
In this project, we explore the application of deep learning techniques for automating the classification of both textual and video data. Leveraging the power of deep neural networks, we aim to create models that can accurately predict the categories of given text and video inputs.

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


# Deep Learning Models
For addressing the text classification challenge, we employed two distinct models: Long Short-Term Memory (LSTM) and Bidirectional LSTM. These models have demonstrated their efficacy in capturing the sequential patterns and contextual information present in textual data, thus enhancing our classification capabilities.

In the realm of video classification, we developed two different models: a 3D Convolutional Neural Network (CNN) and a combination of 2D CNN followed by an LSTM layer. The 3D CNN focuses on analyzing spatiotemporal features within video frames, while the 2D CNN + LSTM architecture effectively captures the temporal patterns and dependencies within video sequences.

![alt](https://github.com/MuhammedGzel/stack-data-structure-applet-visualization/blob/main/app_screenshot.png)
