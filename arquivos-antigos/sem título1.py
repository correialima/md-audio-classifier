'''

arquivo com testes de funções, não considerar

'''

import pandas as pd
import librosa
import numpy as np
import os

# bibliotecas para classificação
from sklearn.metrics import confusion_matrix, classification_report

# bibliotecas para avaliação e exibição dos resultados
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt


# inputs e constantes
PATH_TREINAMENTO = "./TREINAMENTO/"
PATH_VALIDACAO = "./VALIDACAO/"
INTERVALO = 2
rate = 44100

data, fs  = librosa.load('./TREINAMENTO/6acd.wav', None)

dados = data[0:fs*2] / max(abs(data[0:fs*2]))

att = []

stft = np.abs(librosa.stft(dados))
mfccs = np.mean(librosa.feature.mfcc(y=dados, sr=fs, n_mfcc=13),axis=1)
chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=fs),axis=1)
mel = np.mean(librosa.feature.melspectrogram(dados, sr=fs),axis=1)
contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=fs),axis=1)
tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(dados),sr=fs),axis=1)

teste = np.hstack([mfccs, chroma, contrast, tonnetz])

att.append(teste.T)


string = []

teste = 'teste'

string.append(teste[0])

teste[0]

att2 = np.array(string)

att2.T

print(att2.T)

print(att2)

print(len(mfccs))


for i, ini in zip(range(4),range(0, data.shape[0], fs*INTERVALO)):
    dados = data[ini:(ini+fs*2)] # / max(abs(data[ini:(ini+fs*2)]))
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    character_class = file[i]