# Deep_Learning_HDA

Speech recognition using CNN, autoencoder followed by SVM and Shallow Inception

## Get Started

To train the models with your signal data use the Train_Model_ notebooks.

Download the Speech dataset (2.11 GB uncompressed) at http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Once downloaded load the files from the directory where your data is located.

data_dir = 'data/mini_speech_commands/'

To make your feature vectors run the: Calculate the feature vectors of words code snippet, save it in the feature_data folder.

You are now ready to train the models with your words!

## Train_Model_CNN.ipynb

Trains a vanilla CNN on regular spectrograms, MFCC and full feature vectors. Applies dropout and L2 regularization to avoid overfitting. 
Grid search for hyperparameter fine tuning. Save the model.

## Train_Model_Autoencoder.ipynb

Encode the training set images to obtained compressed and smoothed denoised MFCC images for training an SVM. SVM straining. Save the model.

## Train_Model_Inception.ipynb

Train a shallow inception architectur on the word recognition task. Save the model.

## Inference.ipynb

Make inference using the best models on a simple .wav files with a sequence of words.

## Models & Utils

Folders hosting the weights of the models, and utils and model architecture functions respectively

