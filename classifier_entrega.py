#encoding: utf-8
'''
Projeto de Classificador de Áudio

Daniel Pereira Cinalli - 11069711
Rafael Correia de Lima - 21004515

'''
import matplotlib.pyplot as plt
import random
import librosa
import os

# bibliotecas para pré-processamento
import pandas as pd
import numpy as np

# bibliotecas para classificação
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import normalize

# bibliotecas para avaliação e exibição dos resultados
from sklearn.ensemble import RandomForestClassifier


PATH_TREINAMENTO = "./TREINAMENTO/"
PATH_VALIDACAO = "./VALIDACAO/"
PATH_TESTE = "./TESTE/"
SEED = 1000

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = create_set(PATH_TREINAMENTO)
    X_val, y_val = create_set(PATH_VALIDACAO)
    X_test, y_test = create_set(PATH_TESTE)

    X_train = normalize(X_train, axis=0, norm='l2')
    X_val = normalize(X_val, axis=0, norm='l2')
    X_test = normalize(X_test, axis=0, norm='l2')

    classificador = RandomForestClassifier(n_estimators = 75)
    classificador.fit(X_train, y_train)

    # classificação final e avaliação dos resultados

    print("Validação")
    y_pred = classificador.predict(X_val)

    print(classification_report(y_val, y_pred))
    print("Matriz de confusão")
    print(labeled_confusion_matrix(y_val,y_pred))

    # teste
    print("Teste")
    y_pred = classificador.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Matriz de confusão")
    print(labeled_confusion_matrix(y_test, y_pred))

def extract_attributes(dados, fs):
    stft = np.abs(librosa.stft(dados))
    mfccs = np.mean(librosa.feature.mfcc(y = dados, sr = fs, n_mfcc = 13), axis = 1)
    mel = np.mean(librosa.feature.melspectrogram(dados, sr = fs), axis = 1)

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
    

 
if __name__== '__main__':
    main()
    
   
