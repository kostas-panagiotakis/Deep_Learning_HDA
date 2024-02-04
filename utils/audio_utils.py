import cv2
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
import imgaug.augmenters as iaa
import sys

from IPython.display import Image, display
from IPython.display import Audio
from IPython.display import Image
from keras import regularizers
from IPython import display
from scipy.io import wavfile
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation, Input, Dense, LSTM, Attention, Concatenate, Conv2D, BatchNormalization, MaxPooling2D, AveragePooling2D,concatenate, Flatten, Dropout, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from python_speech_features import mfcc, delta, logfbank, fbank
from PIL import Image, ImageDraw, ImageFont
from psutil import virtual_memory

def delta_delta(feat, N=1):
    """
    Compute delta-delta features (acceleration) from a sequence of feature vectors.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta-delta features based on preceding and following N frames of delta features.
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta-delta features. Each row holds 1 delta-delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')

    # Now calculate delta-delta features using the same logic
    delta_delta_feat = delta(feat, N)

    return delta_delta_feat

def example_feature_vectors(file_path, num_words):
    """
    Extracts various acoustic features from an audio file and organizes them into dictionaries.

    Parameters:
    - file_path (str): The path to the audio file.
    - num_words (int): The number of words (not used in the function, consider removing).

    Returns:
    - tuple: A tuple containing signal, sampling rate, and dictionaries of feature vectors.
             Dictionaries include 'signal_dict', 'mfcc_dict', 'delta_dict', 'delta_delta_dict',
             'fbank_dict', and 'full_feature_vector_dict'.
    """
    # Load the audio file
    (rate, sig) = wav.read(file_path)

    # Extract features
    mfcc_feat = mfcc(sig, rate, numcep=13, nfilt=26, nfft=400, appendEnergy=False)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig, rate)
    fbank_feat = fbank_feat[:, :13]
    delta_delta_feat = delta_delta(d_mfcc_feat)

    # Ensure the features have the desired number of time frames
    desired_frames = 99
    features = [mfcc_feat, d_mfcc_feat, delta_delta_feat, fbank_feat]
    features = [np.pad(feature, ((0, max(0, desired_frames - feature.shape[0])), (0, 0)), mode='constant')[:desired_frames, :]
                for feature in features]

    # Extract the desired coefficients for each frame
    num_frames = min(len(feature) for feature in features)
    full_feature_vector = np.concatenate(features, axis=1)[:num_frames, :]

    epsilon = 1e-10  # Small constant to avoid zero energy

    # Extract energy coefficients
    energy_feats = [np.log(np.sum(feature**2, axis=1) + epsilon) for feature in [fbank_feat, d_mfcc_feat, delta_delta_feat]]
    full_feature_vector[:, 40:43] = np.array(energy_feats).T[:num_frames, :]

    # Save the lists of features for the current folder in the dictionaries
    feature_dicts = {'signal_dict': {file_path: np.array([sig])},
                     'mfcc_dict': {file_path: np.array([mfcc_feat])},
                     'delta_dict': {file_path: np.array([d_mfcc_feat])},
                     'delta_delta_dict': {file_path: np.array([delta_delta_feat])},
                     'fbank_dict': {file_path: np.array([fbank_feat])},
                     'full_feature_vector_dict': {file_path: np.array([full_feature_vector])}}
      
    return sig, rate, feature_dicts

def plot_example(signal, rate, fbank_arr, mfcc_arr, delta_arr, delta_delta_arr, file):
    """
    Plots various acoustic features along with the original audio waveform.

    Parameters:
    - signal (numpy.ndarray): The audio waveform.
    - rate (int): Sampling rate of the audio.
    - fbank_arr (dict): Dictionary containing filterbank feature arrays.
    - mfcc_arr (dict): Dictionary containing MFCC feature arrays.
    - delta_arr (dict): Dictionary containing delta feature arrays.
    - delta_delta_arr (dict): Dictionary containing delta-delta feature arrays.
    - file (str): Name of the audio file.

    Returns:
    - None: Displays the plots.
    """
    # Plot the audio waveform
    plt.figure(figsize=(20, 4))

    # Plot original audio waveform
    plt.subplot(1, 5, 1)
    time_points = np.linspace(0, len(signal) / rate, num=len(signal))
    plt.plot(time_points, signal, color='#7E8F7C')  # Change color to a milder tone
    plt.title('Original Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Plot Log Mel-filterbank energy features
    plt.subplot(1, 5, 2)
    plt.imshow(np.transpose(fbank_arr[file]), cmap='viridis', origin='lower', aspect='auto', interpolation='none')
    plt.title('Log Mel-filterbank Energy Features')
    plt.xlabel('Frame')
    plt.ylabel('Filterbank Coefficient')
    plt.colorbar()

    # Plot MFCC features
    plt.subplot(1, 5, 3)
    plt.imshow(np.transpose(mfcc_arr[file]), cmap='viridis', origin='lower', aspect='auto', interpolation='none')
    plt.title('MFCC Features')
    plt.xlabel('Frame')
    plt.ylabel('MFCC Coefficient')
    plt.colorbar()

    # Plot Delta features
    plt.subplot(1, 5, 4)
    plt.imshow(np.transpose(delta_arr[file]), cmap='viridis', origin='lower', aspect='auto', interpolation='none')
    plt.title('Delta Features')
    plt.xlabel('Frame')
    plt.ylabel('Delta Coefficient')
    plt.colorbar()

    # Plot Delta-Delta features
    plt.subplot(1, 5, 5)
    plt.imshow(np.transpose(delta_delta_arr[file]), cmap='viridis', origin='lower', aspect='auto', interpolation='none')
    plt.title('Delta-Delta Features')
    plt.xlabel('Frame')
    plt.ylabel('Delta-Delta Coefficient')
    plt.colorbar()

    # Add an overall title
    plt.suptitle(f'Audio Signal Analysis Example for {file}: Waveform, MFCC, Delta, Delta-Delta, and Filterbank Features', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
    plt.show()


def calculate_feature_vector(data_dir, files, num_words):
    """
    Calculate various acoustic features from audio files in a specified directory.

    Parameters:
    - data_dir (str): The path to the directory containing word folders with audio files.
    - files (list): A list of files in the specified directory.
    - num_words (int): The number of word folders to consider.

    Returns:
    - signal_dict (dict): Dictionary storing raw audio signals for each word.
    - fbank_dict (dict): Dictionary storing Mel-filterbank energy features for each word.
    - mfcc_dict (dict): Dictionary storing Mel-frequency cepstral coefficients for each word.
    - delta_dict (dict): Dictionary storing delta coefficients for each word.
    - delta_delta_dict (dict): Dictionary storing delta-delta coefficients for each word.
    - full_feature_vector_dict (dict): Dictionary storing concatenated feature vectors for each word.
    """

    # Initialize dictionaries to store features
    signal_dict = {}
    mfcc_dict = {}
    delta_dict = {}
    delta_delta_dict = {}
    fbank_dict = {}
    full_feature_vector_dict = {}

    # Set the desired number of time frames
    desired_frames = 99

    # Iterate through each folder
    for file in files[0:num_words]:
        
        file_path = os.path.join(data_dir, file)

        # Check if the item in the directory is a folder
        if os.path.isdir(file_path):
            
            # Get a list of .wav files in the folder
            wav_files = [file for file in os.listdir(file_path) if file.endswith('.wav')]

            # Initialize lists to store features for the current folder
            folder_signals = []
            folder_mfcc_features = []
            folder_delta_features = []
            folder_delta_delta_features = []
            folder_fbank_features = []
            folder_full_feature_vectors = []

            # Iterate through each .wav file in the folder
            for wav_file in wav_files:
                
                # Load the audio file
                wav_path = os.path.join(file_path, wav_file)
                (rate, sig) = wav.read(wav_path)

                # Extract features
                ################################# Compute MFCC features from an audio signal. ################################# 

                # Segment the signal into short frames
                # Calculates the FFT size as a power of two greater than or equal to the number of samples in a single window length.
                # Compute Mel-filterbank energy features from an audio signal.mfcc_feat = mfcc(sig,rate)
                # Take the logarithm of all filterbank energies
                # Take the DCT of the log filterbank energies
                # Keep DCT coefficients 2-13, discard the others
                mfcc_feat = mfcc(sig, rate, numcep=13, nfilt=26, nfft=400, appendEnergy=False)

                ############################### Compute delta features from a feature vector sequence. ########################### 

                # A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
                # trajectories of MFCC coefficients over time
                # Represent local derivatives of MFCC coefficients (“velocity” of s(n))
                d_mfcc_feat = delta(mfcc_feat, 2)

                ######################## Compute log Mel-filterbank energy features from an audio signal. ######################### 

                fbank_feat = logfbank(sig, rate)

                # For ASR, the cepstral coefficients no. 2,3, …,13 are retained, The remaining ones are discarded
                fbank_feat = fbank_feat[:, :13]

                delta_delta_feat = delta_delta(d_mfcc_feat)

                ####################### PADDING ############################
                # Ensure the features have the desired number of time frames
                if mfcc_feat.shape[0] < desired_frames:
                    mfcc_feat = np.pad(mfcc_feat, ((0, desired_frames - mfcc_feat.shape[0]), (0, 0)), mode='constant')
                elif mfcc_feat.shape[0] > desired_frames:
                    mfcc_feat = mfcc_feat[:desired_frames, :]

                if d_mfcc_feat.shape[0] < desired_frames:
                    d_mfcc_feat = np.pad(d_mfcc_feat, ((0, desired_frames - d_mfcc_feat.shape[0]), (0, 0)), mode='constant')
                elif d_mfcc_feat.shape[0] > desired_frames:
                    d_mfcc_feat = d_mfcc_feat[:desired_frames, :]
                    
                if delta_delta_feat.shape[0] < desired_frames:
                    delta_delta_feat = np.pad(delta_delta_feat, ((0, desired_frames - delta_delta_feat.shape[0]), (0, 0)), mode='constant')
                elif delta_delta_feat.shape[0] > desired_frames:
                    delta_delta_feat = delta_delta_feat[:desired_frames, :]

                if fbank_feat.shape[0] < desired_frames:
                    fbank_feat = np.pad(fbank_feat, ((0, desired_frames - fbank_feat.shape[0]), (0, 0)), mode='constant')
                elif fbank_feat.shape[0] > desired_frames:
                    fbank_feat = fbank_feat[:desired_frames, :]
                #############################################################

                # Append features to the lists for the current folder
                folder_signals.append(sig)
                folder_mfcc_features.append(mfcc_feat)
                folder_delta_features.append(d_mfcc_feat)
                folder_delta_delta_features.append(delta_delta_feat)
                folder_fbank_features.append(fbank_feat)
                
                # Extract the desired coefficients for each frame
                num_frames = min(len(mfcc_feat), len(d_mfcc_feat), len(delta_delta_feat), len(fbank_feat))

                # Initialize an array to store the concatenated feature vectors
                full_feature_vector = np.zeros((num_frames, 43))

                # Extract MFCC coefficients
                full_feature_vector[:, :13] = mfcc_feat[:num_frames, :]

                # Extract Delta coefficients
                full_feature_vector[:, 13:26] = d_mfcc_feat[:num_frames, :]

                # Extract Delta-Delta coefficients
                full_feature_vector[:, 26:39] = delta_delta_feat[:num_frames, :]

                epsilon = 1e-10  # Small constant to avoid zero energy

                # Extract signal energy coefficients
                full_feature_vector[:, 40] = np.log(np.sum(fbank_feat[:num_frames, :]**2, axis=1) + epsilon)

                # Extract Delta energy coefficients
                full_feature_vector[:, 41] = np.log(np.sum(d_mfcc_feat[:num_frames, :]**2, axis=1) + epsilon)

                # Extract Delta-Delta energy coefficients
                full_feature_vector[:, 42] = np.log(np.sum(delta_delta_feat[:num_frames, :]**2, axis=1) + epsilon)
                
                folder_full_feature_vectors.append(full_feature_vector)


            # Save the lists of features for the current folder in the dictionaries
            signal_dict[file] = folder_signals
            mfcc_dict[file] = folder_mfcc_features
            delta_dict[file] = folder_delta_features
            delta_delta_dict[file] = folder_delta_delta_features
            fbank_dict[file] = folder_fbank_features
            full_feature_vector_dict[file] = folder_full_feature_vectors

    return signal_dict, fbank_dict, mfcc_dict, delta_dict, delta_delta_dict, full_feature_vector_dict

# Function to convert values in a dictionary to NumPy arrays
def convert_to_arrays(dictionary):
    for key, value in dictionary.items():
        dictionary[key] = np.array(value, dtype=object)
        
# Function to add a new axis to the array in each key of a dictionary
def add_new_axis(dictionary):
    for key, value in dictionary.items():
        dictionary[key] = np.expand_dims(value, axis=-1)

def train_test_creation(Dataset, signal):
    
    # Obtain the classes labels, or words
    y = np.array(list(Dataset.keys()))
    
    # Initialize OneHotEncoder to avoid ordinal relationships
    one_hot_encoder = OneHotEncoder()

    # Reshape y to a 2D array (required by OneHotEncoder)
    y_reshaped = y.reshape(-1, 1)

    # Fit and transform y to one-hot encoded format
    y_one_hot = one_hot_encoder.fit_transform(y_reshaped)

    # Convert one-hot encoded output to an array
    y_encoded = y_one_hot.toarray()

    X = list(Dataset.values())
    sig = list(signal.values())
    
    signal_list = []
    X_list = []
    y_list = []
    for i in range(len(X)):
        for file in range(len(X[i])):
            X_list.append(X[i][file])
            signal_list.append(sig[i][file])
            y_list.append(y_encoded[i])

    # Determine the minimum number of samples for a class
    min_samples = min(len(values) for values in Dataset.values())
    
    # Create a dictionary
    label_indices = {}

    # Iterate through one-hot encoded labels and their corresponding indices
    for index, one_hot_label in enumerate(y_list):
        # Convert one-hot label to tuple for dictionary key
        label_key = tuple(one_hot_label.astype(int))

        # If key exists, append index to the list of values
        if label_key in label_indices:
            label_indices[label_key].append(index)
        # If key does not exist, create a new key-value pair
        else:
            label_indices[label_key] = [index]
    
    # Randomly select min_samples indices for each label
    selected_indices = [random.sample(indices, min_samples) for indices in label_indices.values()]

    # Flatten the list of selected indices
    selected_indices_flat = [index for sublist in selected_indices for index in sublist]

    # Use the selected indices to create a new Paired_Dataset
    Paired_Dataset = [(y_list[i], X_list[i], signal_list[i]) for i in selected_indices_flat]

    # Shuffle Paired_Dataset
    random.shuffle(Paired_Dataset)

    # Extract features and labels from the shuffled dataset
    labels, features, signals = zip(*Paired_Dataset)

    # Assuming 'features' and 'labels' are lists
    features_array = np.array(features)
    labels_array = np.array(labels)
    signals_array = np.array(signals)

    # Split the data into training and testing sets (e.g., 80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.2, random_state=42)
    
    # Split signals_array and labels_array using the same indices
    signals_train, signals_test, labels_train, labels_test = train_test_split(signals_array, labels_array, test_size=0.2, random_state=42)
    
    # Decoding one-hot encoded labels for training and testing sets
    decoded_labels_train = one_hot_encoder.inverse_transform(y_train)
    decoded_labels_test = one_hot_encoder.inverse_transform(y_test)

    return X_train, X_test, y_train, y_test, signals_train, signals_test, y_encoded, one_hot_encoder, decoded_labels_train, decoded_labels_test


def augment_data(X_train, augmentation_percentage=0.3):
    # Determine the number of samples to augment
    num_samples_to_augment = int(len(X_train) * augmentation_percentage)

    # Create an instance of ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Randomly select indices for samples to augment
    indices_to_augment = np.random.choice(len(X_train), num_samples_to_augment, replace=False)

    # Augment selected samples
    augmented_data = []
    for i in range(len(X_train)):
        if i in indices_to_augment:
            augmented_samples = datagen.flow(X_train[i:i + 1], batch_size=1)
            augmented_data.append(augmented_samples[0][0])
        else:
            augmented_data.append(X_train[i])

    return np.array(augmented_data)


def normalize_cache_batch(X_train, X_test, y_train, y_test, cache_batch=False):
    # Flatten each sample in X_train and X_test
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Normalize X_train and X_test using StandardScaler
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train_flat)
    X_test_normalized = scaler.transform(X_test_flat)

    # Reshape back to the original shape if needed
    X_train_normalized = X_train_normalized.reshape(X_train.shape)
    X_test_normalized = X_test_normalized.reshape(X_test.shape)
    IMG_SHAPE = X_train_normalized.shape[1:]
    
    if cache_batch:
        
        # Create TensorFlow Dataset
        batch_size = 64
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_normalized, y_train)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test_normalized, y_test)).batch(batch_size)

        # Optionally cache the datasets to improve performance
        train_dataset = train_dataset.cache()
        test_dataset = test_dataset.cache()
        
        return train_dataset, test_dataset, IMG_SHAPE

    else:
 
        return X_train_normalized, X_test_normalized, y_train, y_test, IMG_SHAPE


def plot_training_history(history):
    """
    Plots the training history, including loss and accuracy.

    Parameters:
    - history (History): The training history obtained from model training.
    """
    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, display_percentages=True):
    """
    Plots a confusion matrix with labels.

    Parameters:
    - y_true (numpy.ndarray): True labels.
    - y_pred (numpy.ndarray): Predicted labels.
    - classes (list): List of class names.
    - display_percentages (bool): Whether to display percentages instead of counts.

    Returns:
    - None: Displays the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    
    if display_percentages:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes, linewidths=.5, cbar=False)
    else:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, linewidths=.5, cbar=False)

    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(fontsize=10, rotation=45, ha="right")
    plt.yticks(fontsize=10, rotation=0)
    plt.show()


def display_memory_usage():
    mem = virtual_memory()
    used_memory = round((mem.total - mem.available) / (1024.0 ** 3), 2)  # Convert to GB
    percent_memory = mem.percent

    print(f"Used Memory: {used_memory} GB")
    print(f"Memory Usage: {percent_memory}%")


# Function to display classification report
def display_classification_report(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n", report)

##################################################         AUTOENCODER        ##################################################

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def preprocess_dataset(dataset, test_size=0.2, random_state=42):
    """
    Preprocesses the given dataset.

    Args:
    - dataset (dict): A dictionary with labels as keys and corresponding spectrograms as values.
    - test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    - random_state (int): Seed used by the random number generator (default is 42).

    Returns:
    - X_train, X_test, Y_train, Y_test, Array_train, Array_test, y_encoded, one_hot_encoder, decoded_labels_train, decoded_labels_test: 
      Processed training and testing data along with decoded labels for training and testing sets.
    """
    
    # Obtain the classes labels, or words
    labels = np.array(list(dataset.keys()))
    
    spectrograms = np.array(list(dataset.values()), dtype=object)
    
    # Initialize OneHotEncoder to avoid ordinal relationships
    one_hot_encoder = OneHotEncoder()

    # Reshape y to a 2D array (required by OneHotEncoder)
    y_reshaped = labels.reshape(-1, 1)

    # Fit and transform y to one-hot encoded format
    y_one_hot = one_hot_encoder.fit_transform(y_reshaped)

    # Convert one-hot encoded output to an array
    y_encoded = y_one_hot.toarray()

    X = []
    Arrays = []
    Y = []

    for w, label in enumerate(spectrograms):
        for i, mfcc in enumerate(label):
            X.append(f'{labels[w]}_{i}')
            Arrays.append(mfcc)
            Y.append(y_encoded[w])

    # Determine the minimum number of samples for a class
    min_samples = min(len(values) for values in dataset.values())
    
    # Create a dictionary
    label_indices = {}

    # Iterate through one-hot encoded labels and their corresponding indices
    for index, one_hot_label in enumerate(Y):
        # Convert one-hot label to tuple for dictionary key
        label_key = tuple(one_hot_label.astype(int))

        # If key exists, append index to the list of values
        if label_key in label_indices:
            label_indices[label_key].append(index)
        # If key does not exist, create a new key-value pair
        else:
            label_indices[label_key] = [index]

    # split the dataset into train and test groups using the package sklearn
    X_train, X_test, Y_train_encoded, Y_test_encoded, Array_train, Array_test = train_test_split(X, Y, Arrays, test_size=test_size, random_state=random_state)

    Array_train = np.array(Array_train)
    Array_test = np.array(Array_test)

    # Specify the amount of padding for each dimension
    pad_width = ((0, 0), (0, 1), (3, 4), (0, 0))

    # Pad the array with zeros
    Array_train = np.pad(Array_train, pad_width, mode='constant', constant_values=0)
    Array_test = np.pad(Array_test, pad_width, mode='constant', constant_values=0)

    # Decoding one-hot encoded labels for training and testing sets
    decoded_labels_train = one_hot_encoder.inverse_transform(Y_train_encoded)
    decoded_labels_test = one_hot_encoder.inverse_transform(Y_test_encoded)

    return X_train, X_test, Y_train_encoded, Y_test_encoded, Array_train, Array_test, y_encoded, one_hot_encoder, decoded_labels_train, decoded_labels_test


def normalize_mfcc(image):
    return tf.cast(image, tf.float32) / 255.

def show_image(x):
    plt.imshow(x)
    
def create_dataset(file_names, img_array, batch_size, shuffle, cache_file=None):
    
    # Create a Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((img_array))
    print("Created dataset from tensor slices")

    # Map the normalize_img function
    dataset = dataset.map(normalize_mfcc, num_parallel_calls=os.cpu_count())
    print("Mapped normalize function")

    # Duplicate data for the autoencoder (input = output)
    py_funct = lambda img: (img, img)
    dataset = dataset.map(py_funct)
    print("Duplicated data for autoencoder (input=output)")

    # Cache dataset
    if cache_file:
        dataset = dataset.cache(cache_file)
        print("Cached dataset")

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(len(file_names))
        print("Shuffled dataset")

    # Repeat the dataset indefinitely
    dataset = dataset.repeat()
    print("Repeated dataset")

    # Batch
    dataset = dataset.batch(batch_size=batch_size)
    print("Batched dataset")

    # Prefetch
    dataset = dataset.prefetch(buffer_size=1)
    print("Prefetched dataset")

    return dataset

# FUNCTION: draws original, encoded and decoded images

def visualize(img, encoder, decoder):
    """
    Arguments:
    img -- original image
    encoder -- trained encoder network
    decoder -- trained decoder network
    """

    code = encoder.predict(img[np.newaxis, :])[0]  # img[np.newaxis, :] is used to add an additional axis
                                                   # Remeber that the model takes as input a 4-dimensional array (?, height, width, channels) where the first dimension
                                                   # is the one related to the mini-batch size. Here our "mini-batch" is composed of a single image
    reco = decoder.predict(code[None])[0]  # img[None] is the same as img[np.newaxis, :]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()
    
    
############################################################ Inception #########################################################

def preprocess_dataset_inception(dataset, test_size=0.2, random_state=42):
    """
    Preprocesses the given dataset.

    Args:
    - dataset (dict): A dictionary with labels as keys and corresponding spectrograms as values.
    - test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    - random_state (int): Seed used by the random number generator (default is 42).

    Returns:
    - X_train, X_test, Y_train, Y_test, Array_train, Array_test: Processed training and testing data.
    """

    # Obtain the classes labels, or words
    labels = np.array(list(dataset.keys()))
    
    spectrograms = np.array(list(dataset.values()), dtype=object)
    
    # Initialize OneHotEncoder to avoid ordinal relationships
    one_hot_encoder = OneHotEncoder()

    # Reshape y to a 2D array (required by OneHotEncoder)
    y_reshaped = labels.reshape(-1, 1)

    # Fit and transform y to one-hot encoded format
    y_one_hot = one_hot_encoder.fit_transform(y_reshaped)

    # Convert one-hot encoded output to an array
    y_encoded = y_one_hot.toarray()


    X = []
    Y = []

    for w, label in enumerate(spectrograms):
        for i, mfcc in enumerate(label):
            X.append(mfcc)
            Y.append(y_encoded[w])

    # split the dataset into train and test groups using the package sklearn
    X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    # Add a channel axis at the end for Array_train
    X_train = X_train[..., np.newaxis]

    # Add a channel axis at the end for Array_test
    X_test = X_test[..., np.newaxis]

    # Specify the amount of padding for each dimension
    pad_width = ((0, 0), (0, 1), (3, 4), (0, 0))

    # Pad the array with zeros
    X_train = np.pad(X_train, pad_width, mode='constant', constant_values=0)
    X_test = np.pad(X_test, pad_width, mode='constant', constant_values=0)

    # Decoding one-hot encoded labels for training and testing sets
    decoded_labels_train = one_hot_encoder.inverse_transform(Y_train)
    decoded_labels_test = one_hot_encoder.inverse_transform(Y_test)

    return X_train, X_test, Y_train, Y_test, y_encoded, one_hot_encoder, decoded_labels_train, decoded_labels_test

############################################################ Inference #########################################################


def inference_sig_to_mfcc(output_folder, intervals):

    desired_frames = 99 

    # Iterate through the saved subfiles and store each audio signal
    for i in range(1, len(intervals) + 1):
        subfile_path = os.path.join(output_folder, f'subfile_{i}.wav')
        subfile_data, subfile_rate = librosa.load(subfile_path, sr=None)

        # Get a list of .wav files in the folder
        wav_files = [file for file in os.listdir(output_folder) if file.endswith('.wav')]

        # Initialize lists to store features for the current folder
        folder_signals = []
        folder_mfcc_features = []

        # Iterate through each .wav file in the folder
        for wav_file in wav_files:

            # Load the audio file
            wav_path = os.path.join(output_folder, wav_file)
            (rate, sig) = wav.read(wav_path)

            # Extract features
            ################################# Compute MFCC features from an audio signal. ################################# 

            # Segment the signal into short frames
            # Calculates the FFT size as a power of two greater than or equal to the number of samples in a single window length.
            # Compute Mel-filterbank energy features from an audio signal.mfcc_feat = mfcc(sig,rate)
            # Take the logarithm of all filterbank energies
            # Take the DCT of the log filterbank energies
            # Keep DCT coefficients 2-13, discard the others
            mfcc_feat = mfcc(sig, rate, numcep=13, nfilt=26, nfft=400, appendEnergy=False)

            ####################### PADDING ############################
            # Ensure the features have the desired number of time frames
            if mfcc_feat.shape[0] < desired_frames:
                mfcc_feat = np.pad(mfcc_feat, ((0, desired_frames - mfcc_feat.shape[0]), (0, 0)), mode='constant')
            elif mfcc_feat.shape[0] > desired_frames:
                mfcc_feat = mfcc_feat[:desired_frames, :]

            # Append features to the lists for the current folder
            folder_signals.append(sig)
            folder_mfcc_features.append(mfcc_feat)

    return folder_signals, folder_mfcc_features, wav_files