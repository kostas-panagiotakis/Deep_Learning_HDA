import pickle
import random
import joblib
import librosa
import scipy.signal
import wave
import scipy.io.wavfile as wav
import os
import pathlib
import psutil
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from IPython.display import Image, display
from IPython.display import Audio
from IPython.display import Image
from keras import regularizers
from IPython import display
from scipy.io import wavfile
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, LSTM, Attention, Concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from python_speech_features import mfcc, delta, logfbank, fbank
from PIL import Image, ImageDraw, ImageFont
from psutil import virtual_memory

from utils.audio_utils import *

def CNN_model_simple_batch(train_dataset, test_dataset, one_hot_encoder, num_classes, N_Epochs, inp_shape):
    # Define the CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=inp_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_dataset, epochs=N_Epochs, validation_data=test_dataset)

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Predict on test dataset
    y_true = np.concatenate([np.argmax(y, axis=1) for x, y in test_dataset], axis=0)
    y_pred = model.predict(test_dataset).argmax(axis=1)

    # Classification Report
    class_report = classification_report(y_true, y_pred, target_names=one_hot_encoder.categories_[0])
    print("Classification Report:")
    print(class_report)

    # Generate confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix to percentages
    confusion_mat_norm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=one_hot_encoder.categories_[0], yticklabels=one_hot_encoder.categories_[0])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Normalized)')
    plt.show()

    # Memory Allocation and Usage
    process = psutil.Process(os.getpid())
    print(f"Memory Allocation: {process.memory_info().rss / (1024 ** 2):.2f} MB")
    print(f"Memory Usage: {psutil.virtual_memory().percent:.2f}")

    return model

def CNN_model_with_dropout_batch(train_dataset, test_dataset, one_hot_encoder, num_classes, N_Epochs, inp_shape, dropout_rate=0.2, l2_regularization=0.01, learning_rate=0.001):
    
    # Define the CNN model with dropout and regularization
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_regularization), input_shape=inp_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_dataset, epochs=N_Epochs, validation_data=test_dataset)

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Predict on test dataset
    y_true = np.concatenate([np.argmax(y, axis=1) for x, y in test_dataset], axis=0)
    y_pred = model.predict(test_dataset).argmax(axis=1)

    # Classification Report
    class_report = classification_report(y_true, y_pred, target_names=one_hot_encoder.categories_[0])
    print("Classification Report:")
    print(class_report)

    # Generate confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix to percentages
    confusion_mat_norm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=one_hot_encoder.categories_[0], yticklabels=one_hot_encoder.categories_[0])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Normalized)')
    plt.show()

    # Memory Allocation and Usage
    process = psutil.Process(os.getpid())
    print(f"Memory Allocation: {process.memory_info().rss / (1024 ** 2):.2f} MB")
    print(f"Memory Usage: {psutil.virtual_memory().percent:.2f}")

    return model

##################################################         AUTOENCODER        ##################################################


def build_deep_autoencoder(img_shape, code_size):
    """
    Arguments:
    img_shape_x -- size of the input layer
    code_size -- the size of the hidden representation of the input (code)

    Returns:
    encoder -- keras model for the encoder network
    decoder -- keras model for the decoder network
    """

    # encoder
    ### START CODE HERE ###
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.Input(img_shape))

    encoder.add(layers.Conv2D(32, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Conv2D(64, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Conv2D(128, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Conv2D(256, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(code_size))
    ### END CODE HERE ###

    # decoder
    ### START CODE HERE ###
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.Input((code_size,)))

    decoder.add(layers.Dense(6 * 1 * 256, activation='elu'))
    decoder.add(layers.Reshape((6, 1, 256)))
    decoder.add(layers.Conv2DTranspose(128, (3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=2, activation='elu', padding='valid'))
    decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(layers.Conv2DTranspose(1, (3, 3), strides=2, activation=None, padding='same'))
    ### END CODE HERE ###

    return encoder, decoder


