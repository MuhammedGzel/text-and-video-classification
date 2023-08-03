from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Embedding
from keras.utils import plot_model


def create_lstm_model(max_features, max_len, num_classes):
    model = Sequential()
    model.add(Embedding(max_features, 300, input_length=max_len))
    model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, show_shapes=True, to_file="lstm_model.png")
    return model


def create_bidirectional_lstm_model(max_features, max_len, num_classes):
    model = Sequential()
    model.add(Embedding(max_features, 300, input_length=max_len))
    model.add(Bidirectional(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, show_shapes=True, to_file="bidirectional_lstm_model.png")
    return model

