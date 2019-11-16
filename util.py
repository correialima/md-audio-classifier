
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

INTERVALO = 2

def create_set(path):
    X = []
    Y = []
    
    for file in sorted(os.listdir(path)):
        data, fs  = librosa.load(path+file, None)
        for i, ini in zip(range(4),range(0, data.shape[0], fs*INTERVALO)):
            Y.append(file[i])
            dados = data[ini:(ini+fs*2)] / max(abs(data[ini:(ini+fs*2)]))
            stft = np.abs(librosa.stft(dados))
            mfccs = np.mean(librosa.feature.mfcc(y=dados, sr=fs, n_mfcc=13),axis=1)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=fs),axis=1)
            mel = np.mean(librosa.feature.melspectrogram(dados, sr=fs),axis=1)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=fs),axis=1)
            centroid = np.mean(librosa.feature.spectral_centroid(dados,S=stft))
            #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(dados),sr=fs),axis=1)
            
            att = np.hstack([mfccs,chroma,contrast,mel,centroid])
            X.append(att)

    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y
    
def labeled_confusion_matrix(y_val,y_pred):
    pred_labels = np.unique(y_pred)
    cm = pd.DataFrame(confusion_matrix(y_val,y_pred, labels = pred_labels), index = pred_labels, columns = pred_labels)

    return cm
