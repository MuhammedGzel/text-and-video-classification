import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


def plot_training_graph(history, model_name):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)
    ax1.set_title(model_name + ' Modeli Eğitim Doğruluk Grafiği')
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='test')
    ax1.set_ylabel('Accuracy(Doğruluk)')
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('Epoch(Tur)')
    ax1.legend(['Train', 'Val'])

    ax2.set_title(model_name + ' Modeli Eğitim Kayıp Grafiği')
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='test')
    ax2.set_ylabel('Loss(Kayıp)')
    max_loss = max(history.history['loss'] + history.history['val_loss'])
    ax2.set_ylim([0, np.ceil(max_loss)])
    ax2.set_xlabel('Epoch(Tur)')
    ax2.legend(['Train', 'Val'])
    plt.show()


def get_actual_predicted_labels(dataset, model):
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)
    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)
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
