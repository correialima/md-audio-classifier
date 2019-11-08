import glob
import pandas as pd
import librosa
import numpy as np
import os

import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

PATH_TREINAMENTO = "./TREINAMENTO/"
PATH_VALIDACAO = "./VALIDACAO/"
INTERVALO = 2

def normalizeColumns(df, colnames):
    
    main_max_scaler = preprocessing.MinMaxScaler()
    
    for colname in colnames:
        col = df[colname].values.reshape(-1, 1)
        col_scaled = main_max_scaler.fit_transform(col)
        df[colname] = col_scaled
        
    return df

def getRealValues(filename):
    return filename[0:4]

def getFolderContents(path):
    return [os.path.basename(x) for x in glob.glob(f"{path}*")]

def getAudioSegments(filename):
    data, fs = librosa.load(filename, None)
    duracao_total = data.shape[0]/fs
    dados_p_seg = {}

    for i, ini in enumerate(range(0, data.shape[0], fs*INTERVALO)):
        dados_p_seg[i] = pd.Series(data[ini:(ini + fs*INTERVALO)])
        
    return dados_p_seg
    
def createDF(path):
    filenames = getFolderContents(path)
    features = ["Real", "MediaMFCC", "MedianaMFCC", "MaxMFCC", "MinMFCC"]
    df = pd.DataFrame(columns = features)
    
    for filename in filenames:
        
        realValues = getRealValues(filename)
        dados_p_seg = getAudioSegments(f"{path}{filename}")
        
        for real, seg in zip(realValues, dados_p_seg.values()):
            mfcc = librosa.feature.mfcc(y = seg.to_numpy())
            media = np.mean(mfcc)
            mediana = np.median(mfcc)
            minm = np.min(mfcc)
            maxm = np.max(mfcc)
            df = df.append({"Real": real, "MediaMFCC": media, "MedianaMFCC": mediana, "MaxMFCC": maxm, "MinMFCC": minm}, ignore_index = True)
  
    return df

def main():
    df_treinamento = createDF(PATH_TREINAMENTO)
    df_validacao = createDF(PATH_VALIDACAO)
    
    normalizeColumns(df_treinamento,["MediaMFCC", "MedianaMFCC", "MaxMFCC", "MinMFCC"])
    
    normalizeColumns(df_validacao,["MediaMFCC", "MedianaMFCC", "MaxMFCC", "MinMFCC"])
    
    
    X_train = df_treinamento[["MediaMFCC", "MedianaMFCC", "MaxMFCC", "MinMFCC"]].to_numpy()
    X_test  = df_validacao[["MediaMFCC", "MedianaMFCC", "MaxMFCC", "MinMFCC"]].to_numpy()
    y_train = df_treinamento[["Real"]].to_numpy().ravel()
    y_test  = df_validacao[["Real"]].to_numpy().ravel()

    clf = RandomForestClassifier(n_estimators = 30)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    print(sum(y_pred == y_test)/len(y_pred))
    
if __name__ == '__main__':
    main()




