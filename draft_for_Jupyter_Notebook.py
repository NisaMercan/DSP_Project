#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### This is a Digital Signal Processing Laboratory project designed and coded by 

#    F. Nisa Mercan
#    Sena Gücük
#    Kutay Acar

# In this project we extracted features of two classes of sound:
# Rain and Boiling oil by making use of librosa library to extract features
# and sci-kit learn to build Support Vector Machines.

## Extract the WAV files
get_ipython().system('unzip /content/drive/MyDrive/dataset.zip')


import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sgnl
from scipy.fftpack import fft, ifft
import scipy.io
from scipy.io.wavfile import read     # to open and return Fs of the WAV file
from IPython.lib.display import Audio # to create audio files 
import sklearn               # for machine learning and statistical modeling including classification
from sklearn.svm import SVC  # support vector machine
from sklearn.model_selection import train_test_split # to split the matrices into train and subset
from sklearn.model_selection import GridSearchCV     # to help to loop through predefined hyperparameters and model the training set
from sklearn.metrics import classification_report    # to build a text report that contains the main classifications
import joblib
#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')


############### Reading Files ###############
def readAudio(s, ad):
    Fs, x = read(s)
    xn = x[:,1]
    print(f'{ad} shape: ', x.shape)
    print(f'{ad} sampling frequency: ', Fs) 
    return Fs, xn

Fs_ygm4, ygm4 = readAudio("/content/dataset/002-rain/b10_rain_4.wav", "Rain4")
Fs_kzrt4, kzrt4 = readAudio("/content/dataset/001-fry/b10_fry_4.wav", "Fry4")


############### Plotting Signals ###############
def plotAudio(xn, ad, i):
    N = len(xn)
    n = np.arange(0,N)
    
    plt.subplots(figsize=(4, 4))
    plt.tight_layout()
    
    plt.plot(n,xn)
    plt.xlabel(f'n (sample)')
    plt.title(f'{ad} Isareti')
    plt.show()

plotAudio(ygm4, "YAGMUR4", 1)
plotAudio(kzrt4, "KIZARTMA4", 2)


############### Amplitude and Phase ###############
def FTAudio_abs(N, xn, ad):
    w = np.arange(-np.pi,np.pi,2*np.pi/N)
    xw = np.fft.fftshift(np.fft.fft(xn,N)/N)

    plt.subplots(figsize=(4, 4))
    plt.plot(w/np.pi,abs(xw)) 
    plt.xlabel('$\omega / \pi$')
    plt.ylabel(f'$|{ad}(\omega)|$')    
    plt.title(f'{ad} Isareti Genligi')
    plt.show()    
    return xw

def FTAudio_pha(N, xn, ad):
    w = np.arange(-np.pi,np.pi,2*np.pi/N)
    xw = np.fft.fftshift(np.fft.fft(xn,N)/N)

    plt.subplots(figsize=(4, 4))
    plt.plot(w/np.pi,np.angle(xw)) 
    plt.xlabel('$\omega / \pi$')
    plt.ylabel(f'$|{ad}(\omega)|$')    
    plt.title(f'{ad} Isareti Fazi')
    plt.show()

ygmw4 = FTAudio_abs(len(ygm4), ygm4, "YAGMUR4")
kzrtw4 = FTAudio_abs(len(kzrt4), kzrt4,"KIZARTMA4")


############### Audio Display ###############
print('YAGMUR4 işareti için:')
display(Audio(ygm4, rate=Fs_ygm4))

print('KIZARTMA4 işareti için:')
display(Audio(kzrt4, rate=Fs_kzrt4))



############### Required Libraries and Packages ###############
get_ipython().system('pip install librosa')
get_ipython().system('pip install sounddevice  # for play and record np.arrays')
get_ipython().system('sudo apt-get install libportaudio2   # portable audio I/O library')

from matplotlib.pyplot import specgram # to plot a spectrogram

import soundfile as sf                # to read and write files
import sounddevice as sd              # for play and record np.arrays
import librosa                        # for music and sound analysis
import librosa.display                # for plotting the amplitude
import sklearn                        # for SVM
from sklearn.preprocessing import Normalizer # to normalize samples to unit norm 
from os.path import dirname, join as pjoin # for functions on multiple pathnames  
from scipy.io import wavfile

import code
import glob  
import os    # to interact with OS
import queue # to operate like a stack
from tqdm.notebook import tqdm # to show a progress meter bar


############### Feature Extraction ###############
# path_to_dataset: takes one parameter that shows the path
def features_to_numpy(path_to_dataset): 
  labels = [] # fry or rain
  features = np.empty((0,193)) # 0 to 193 dimensions
  print("Starting walking in the directory of dataset for feature extraction...")
  for root, dirs, files in tqdm(os.walk(path_to_dataset)): # all dataset files
      for file in tqdm(files): # progress bar
          if file.endswith(".wav"): # file extension
             file_name=os.path.join(root, file) # concatenates path components
             
             X, sample_rate = sf.read(file_name, dtype='float32') # read file
             
             # convert X array to vector 
             if X.ndim > 1: X = X[:,0]
             X = X.T
             
            #citation: we got feature extraction from the open source code: https://github.com/MasazI/audio_classification

             ## Short Time FT: 
             # represents the signal in the freq domain by using DFT and
             # extract frequencies of harmonics in the audio signal
             stft = np.abs(librosa.stft(X)) 
             
             ## Mel Frequency Cepstral Coefficients
             # y : audio time series
             # sr: sample rate of y
             # n_mffc: number of MFCCs to return
             mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
             
             # Mel Scaled Spectrogram
             # computes a mel-scaled spectrogram from audio series
             mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

             ## Spectral Contrast
             # considers the spectral peak and their difference in each
             # frequency subband and computes the spectral contrast
             contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
             
             ## Tonnetz
             # contains harmonic content of the audio signal
             # extracts the harmonic elements from the audio 
             tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
             
             ## Chromatagram
             # computes a chromagram
             # chromagram is a transformation of a signal's time-frequency
             # into a temporally varying pitch
             chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

             # stack the algorithm arrays into one single array both horizontally and vertically 
             ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz]) 
             features = np.vstack([features,ext_features])
          
             # label the sounds
             if os.path.basename(os.path.dirname(file_name)) == "001-fry":
               labels.append(0) # fry sounds - 0
             else:
                labels.append(1) # rain sounds - 1
  labels = np.array(labels) 
  np.save('features.npy', features) # saves the array into bin file as .npy 
  np.save('label.npy', labels) # saves the array into bin file as .npy


############### Load Data From Numpy File ###############
X =  np.load('features.npy')
y =  np.load('label.npy').ravel()

#len(X)

# Split data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = SVC() 
model.fit(X_train, y_train)
  
# print the prediction results
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

#https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
# grid search is implemented from this code

from sklearn.model_selection import GridSearchCV
  
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

  
# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)  


# class sklearn.svm.SVC(*, C=1.0, gamma='scale', kernel='rbf', decision_function_shape='ovr', degree=3,
#                       max_iter=- 1, probability=False, random_state=None, shrinking=True, tol=0.001)

model = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
model.fit(X_train, y_train)

# print prediction results
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

def differentiate(path_to_wav):
  X, sample_rate = sf.read(path_to_wav, dtype='float32')
  features, labels = np.empty((0,193)), np.empty(0)
  X = X[1:len(X),0 ]
  
  stft = np.abs(librosa.stft(X))
  db = librosa.amplitude_to_db(stft,ref=np.max)
  mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
  mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
  contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
  tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
  chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

  plt.subplots(figsize=(7, 10))
  
# STFT
  plt.subplot(3,2,1)
  librosa.display.specshow(db, sr=sample_rate, y_axis='log', x_axis='time')
  #plt.plot(stft)
  plt.title('Short Time FT')

  plt.subplot(3,2,2)
  librosa.display.specshow(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)) 
  plt.colorbar()
  plt.tight_layout()


  plt.plot(mfccs)
  plt.title('Mel Frequency Cepstral Coefficients')

  
  plt.subplot(3,2,3)
  #plt.plot(mel)
  S = librosa.feature.melspectrogram(X, sr=sample_rate)
  S_DB = librosa.power_to_db(S, ref=np.max)
  librosa.display.specshow(S_DB, sr=sample_rate, x_axis='time', y_axis='mel');
  plt.colorbar(format='%+2.0f dB');
  plt.title('MEL')

  plt.subplot(3,2,4)
  #plt.plot(contrast)
  plt.imshow(normalize(librosa.feature.spectral_contrast(S=stft, sr=sample_rate), axis=1), aspect='auto', origin='lower', cmap='coolwarm')
  plt.title('Spectral Contrast')

  plt.subplot(3,2,5)
  #plt.plot(tonnetz)
  plt.imshow(normalize(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate), axis=1), aspect='auto', origin='lower', cmap='coolwarm')
  plt.title('Tonnetz')

  plt.subplot(3,2,6)
  #plt.plot(chroma)
  plt.imshow(normalize(librosa.feature.chroma_stft(S=stft, sr=sample_rate)), aspect='auto', origin='lower', cmap='coolwarm')
  plt.title('Chromagram')

  ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
  features = np.vstack([features,ext_features])
  deneme_test = np.array(features)
  if model.predict(deneme_test) == 0:
    print("this is a fry sound")
  elif model.predict(deneme_test) == 1:
    print("this is a rain sound")
    
    
    
############### Differentiate Two Different Sounds ###############    
differentiate("/content/173303__kvgarlic__baconfryingwinterearly2013.wav")
differentiate("/content/495379__tosha73__strong-rain.wav")



filename = 'dspclassifier.sav'
joblib.dump(model, filename) # object to a file

