import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from models import create_3d_cnn_model, create_2d_cnn_plus_lstm_model
from frame_sampling import FrameGenerator
from evaluate_video_classification import plot_training_graph, get_actual_predicted_labels, plot_classification_metrics, plot_confusion_matrix

num_clases = 25
n_frames = 15
batch_size = 4
HEIGHT = 224
WIDTH = 224

trainPath = pathlib.Path('Dataset/train')
valPath = pathlib.Path('Dataset/val')
testPath = pathlib.Path('Dataset/test')

output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(trainPath, n_frames, training=True),
                                          output_signature=output_signature)
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(valPath, n_frames), output_signature=output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(testPath, n_frames), output_signature=output_signature)
test_ds = test_ds.batch(batch_size)

input_shape = (None, n_frames, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
model = create_2d_cnn_plus_lstm_model(input, num_clases)

frames, label = next(iter(train_ds))
model.build(frames)
keras.utils.plot_model(model, expand_nested=True, dpi=100, show_shapes=True)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

history = model.fit(x=train_ds, epochs=20, validation_data=val_ds)
model.save("Networks/2D CNN+LSTM/video_net_2d_cnn_plus_lstm.h5")
plot_training_graph(history, "2D CNN+LSTM")

model.evaluate(test_ds, return_dict=True)
fg = FrameGenerator(trainPath, n_frames, training=True)
labels = list(fg.class_ids_for_name.keys())
actual, predicted = get_actual_predicted_labels(test_ds, model)
plot_confusion_matrix(actual, predicted, labels, "2D CNN+LSTM")
plot_classification_metrics(actual, predicted, labels, "2D CNN+LSTM")
