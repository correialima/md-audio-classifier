'''
Projeto de Classificador de Áudio

Daniel Pereira Cinalli - 
Rafael Correia de Lima - 21004515

'''

# bibliotecas para pré-processamento
import pandas as pd
import librosa
import numpy as np
import os

# bibliotecas para classificação
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import normalize

# bibliotecas para avaliação e exibição dos resultados
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# inputs e constantes
PATH_TREINAMENTO = "./TREINAMENTO/"
PATH_VALIDACAO = "./VALIDACAO/"
INTERVALO = 2
rate = 44100


# funções criadas

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
            #mel = np.mean(librosa.feature.melspectrogram(dados, sr=fs),axis=1)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=fs),axis=1)
            #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(dados),sr=fs),axis=1)
            
            att = np.hstack([mfccs,chroma,contrast])
            X.append(att)

    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

X_train, y_train = create_set(PATH_TREINAMENTO)
X_val, y_val = create_set(PATH_VALIDACAO)

X_train = normalize(X_train, axis=0, norm='l2')
X_val = normalize(X_val, axis=0, norm='l2')



classificador = RandomForestClassifier(n_estimators=1000)
classificador.fit(X_train, y_train)


classificador = KNeighborsClassifier(n_neighbors=3)
classificador.fit(X_train, y_train)


# classificação final e avaliação dos resultados

y_pred = classificador.predict(X_val)

print(classification_report(y_val, y_pred))



