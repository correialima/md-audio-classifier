import librosa
import pandas as pd
import os
import sys
'''
parte 1

iteração através dos arquivos da pasta, criando dataframe para análise

'''
train_path = './TREINAMENTO/'
val_path = './VALIDACAO/'

dados = []

for file in sorted(os.listdir(train_path)):
    data, fs  = librosa.load(train_path+file, None)
    #duracao_total = data.shape[0]/fs
    #intervalo = 2
    dados_p_seg = []
    for i, ini in zip(range(4),range(0, data.shape[0], fs*intervalo)):     
        dados.append([data[ini:(ini+fs*2)],file.split('.')[0],file[i]])

train_set = pd.DataFrame(dados, columns = ['dados','source_file','class'])

print(train_set)