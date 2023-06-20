import keras
from keras import Model
from keras.layers import Conv3D, BatchNormalization, MaxPooling3D, Flatten, Dense, Dropout, TimeDistributed, \
    GlobalAveragePooling2D, LSTM
from keras.applications import MobileNetV2


def create_3d_cnn_model(input, num_classes):
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(input)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)
    return model


def create_2d_cnn_plus_lstm_model(input, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = TimeDistributed(base_model)(input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(64)(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)
    return model
