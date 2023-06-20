import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


def plot_training_graph(history, model_name):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(model_name + ' Modeli Eğitim Doğruluk Grafiği')
    plt.ylabel('Accuracy(Doğruluk)')
    plt.xlabel('Epoch(Tur)')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name + ' Modeli Eğitim Kayıp Grafiği')
    plt.ylabel('Loss(Kayıp)')
    plt.xlabel('Epoch(Tur)')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


def get_actual_predicted_labels(dataset, labels, model):
    predicted = model.predict(dataset)
    predicted = np.argmax(predicted, axis=1)
    actual = np.argmax(labels, axis=1)
    return actual, predicted


def plot_confusion_matrix(actual, predicted, labels, model_name):
    cm = tf.math.confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    sns.set(rc={'figure.figsize': (12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title(model_name + ' Modeli Test Sonucu Hata Matrisi')
    ax.set_xlabel('Tahmin Edilen Sınıf')
    ax.set_ylabel('Gerçek Sınıf')
    plt.show()


def plot_classification_metrics(actual, predicted, labels, model_name):
    report = classification_report(actual, predicted, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.drop(['accuracy', 'macro avg', 'weighted avg'], inplace=True)

    heatmap_data = report_df[['precision', 'recall', 'f1-score']]
    ax = sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.2f')
    ax.set_title(model_name + ' Modeli Test Sonucu Performans Metrikleri')
    ax.set_xticklabels(['Precision (Hassasiyet)', 'Recall (Geri Çağırma)', 'F1-Score (F1-Skoru)'])
    ax.set_xlabel('Metrikler')
    ax.set_ylabel('Sınıflar')
    plt.show()
