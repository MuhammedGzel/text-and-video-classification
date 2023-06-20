from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras_preprocessing.text import Tokenizer

from clear_text_dataset import prepare_data, clear_devamini_oku


def calculate_data_numbers_and_percentages():
    df = pd.read_csv("Dataset/dataset.csv")
    data_counts = df["category"].value_counts()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=data_counts.index, y=data_counts.values)
    plt.xticks(rotation=90)
    plt.xlabel("Sınıflar")
    plt.ylabel("Veri Sayısı")
    plt.title("Sınıflar İçin Veri Sayıları")
    plt.show()
    plt.figure(figsize=(10, 10))
    plt.pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
    plt.axis('equal')
    plt.title("Veri Sayılarının Sınıflara Göre Yüzdelik Dağılımı")
    plt.show()


def calculate_embedding_parameters(csv_file, min_frequency=5):
    df = pd.read_csv(csv_file)
    df['text'] = df['text'].apply(clear_devamini_oku)
    df['text'] = df['text'].apply(prepare_data)
    text_data = df['text'].tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data)
    word_counts = Counter(tokenizer.word_counts)
    frequent_words = [word for word, count in word_counts.items() if count >= min_frequency]
    vocabulary_size = len(frequent_words) + 1
    max_len = max(len(text.split()) for text in text_data)
    print("Frekansı 5'den büyük eşsiz kelime sayısı:", vocabulary_size, "En uzun metnin uzunluğu:", max_len)


calculate_embedding_parameters("Dataset/dataset.csv")
