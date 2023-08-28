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


# Datasets Used
For training our text classification models, we utilized the TC32 dataset. This dataset offers a diverse range of textual data, enabling us to develop models that can effectively categorize text inputs into relevant classes.

When addressing video classification, we worked with a subset of 25 classes from the UCF-101 dataset. This subset was chosen to streamline the training process and ensure efficient model development. The UCF-101 dataset is renowned for its broad collection of action videos, which serve as a suitable foundation for our video classification tasks.


# User Interface Development
To facilitate the utilization of our trained models, we designed a user-friendly interface using PyQt5. This interface enables users to interact with our models seamlessly, making class predictions for various types of input data including audio, video, and text.

Additionally, our interface supports audio and video files for text classification tasks, further enhancing the versatility and usability of our solution. Users can intuitively provide input data through the interface and receive accurate predictions from our deep learning models.


<br />
<br />

<img src=https://github.com/MuhammedGzel/text-and-video-classification/blob/master/images/video_classification_screen.png width="850" height="500">
<br />
<img src=https://github.com/MuhammedGzel/text-and-video-classification/blob/master/images/text_classification_from_text_screen.png width="850" height="500">
<br />
<img src=https://github.com/MuhammedGzel/text-and-video-classification/blob/master/images/text_classification_from_media_screen.png width="850" height="500">

