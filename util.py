
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

from config import INTERVALO

def extract_attributes(dados, fs):
    #_, dados = librosa.effects.hpss(y = dados)
    #dados, _ = librosa.effects.trim(y = dados, top_db = 20)
    stft = np.abs(librosa.stft(dados))
    mfccs = np.mean(librosa.feature.mfcc(y = dados, sr = fs, n_mfcc = 13), axis = 1)
    #chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = fs), axis = 1)
    mel = np.mean(librosa.feature.melspectrogram(dados, sr = fs), axis = 1)
    #contrast = np.mean(librosa.feature.spectral_contrast(S = stft, sr = fs), axis = 1)
    #centroid = np.mean(librosa.feature.spectral_centroid(dados, S = stft))
    #zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y = dados))
    #tonnetz = np.mean(librosa.feature.tonnetz(y = librosa.effects.harmonic(dados), sr = fs), axis = 1)
    
    att = np.hstack([
        mfccs,
        mel
        ])
    return att

def getAudioSegments(filename):
    data, fs = librosa.load(filename, None)
    duracao_total = data.shape[0]/fs
    dados_p_seg = {}

    for i, ini in enumerate(range(0, data.shape[0], fs * INTERVALO)):
        dados_p_seg[i] = data[ini:(ini + fs * 2)] / max(abs(data[ini:(ini + fs * 2)]))
        
    return dados_p_seg, fs

def loadAudio(filename):
    return librosa.load(filename, None)

def getFilenames(path):
    return sorted(os.listdir(path))    

def create_set(path):
    X = []
    Y = []
    
    for filename in getFilenames(path):
        dados_segmentados, fs  = getAudioSegments(path+filename)
        for i, dados in zip(range(4),dados_segmentados.values()):
            Y.append(filename[i])
            att = extract_attributes(dados, fs)
            X.append(att)

    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

def labeled_confusion_matrix(y_val,y_pred):
    pred_labels = np.unique(y_pred)
    cm = pd.DataFrame(confusion_matrix(y_val,y_pred, labels = pred_labels), index = pred_labels, columns = pred_labels)

    return cm
   
   
def rearrange(y, n):
    """Recebe ndarray de chars e retorna lista de strings concatenando de n em n"""
    y = y.reshape((len(y)//n, n))
    return [''.join(row) for row in y]
        
    
    
