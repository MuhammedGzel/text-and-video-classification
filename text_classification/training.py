import pandas as pd
from clear_text_data import clear_devamini_oku, prepare_data
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import pickle
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from models import create_lstm_model, create_bidirectional_lstm_model
from evaluate_text_classification import get_actual_predicted_labels, plot_classification_metrics, plot_confusion_matrix, \
    plot_training_graph

df = pd.read_csv('dataset/dataset.csv')
df['text'] = df['text'].apply(clear_devamini_oku)
df['text'] = df['text'].apply(prepare_data)

X_train, X_val_test, y_train, y_val_test = train_test_split(df['text'], df['category'], test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

max_len = 164
max_features = 92997
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)

with open('networks/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_val_enc = encoder.transform(y_val)
y_test_enc = encoder.transform(y_test)

num_classes = len(np.unique(y_train_enc))
y_train_enc = tf.keras.utils.to_categorical(y_train_enc, num_classes)
y_val_enc = tf.keras.utils.to_categorical(y_val_enc, num_classes)
y_test_enc = tf.keras.utils.to_categorical(y_test_enc, num_classes)

model = create_bidirectional_lstm_model(max_features, max_len, num_classes)
history = model.fit(X_train_pad, y_train_enc, validation_data=(X_val_pad, y_val_enc), epochs=7, batch_size=448)
model.save("networks/bidirectional_lstm/text_net_bidirectional_lstm.h5")
plot_training_graph(history, " İki Yönlü lstm")

actual, predicted = get_actual_predicted_labels(X_test_pad, y_test_enc, model)
plot_confusion_matrix(actual, predicted, encoder.classes_, "İki Yönlü lstm")
plot_classification_metrics(actual, predicted, encoder.classes_, "İki Yönlü lstm")
