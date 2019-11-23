import pandas as pd
import librosa
import numpy as np
import os

'''
parte 1

iteração através dos arquivos da pasta, criando dataframe para análise

'''
train_path = './TREINAMENTO/'
val_path = './VALIDACAO/'
intervalo = 2


def create_feaures(data_set):
    return

def create_set(path):
    df = pd.DataFrame(columns = ['dados','source_file','class'])
    for file in sorted(os.listdir(path)):
        data, fs  = librosa.load(path+file, None)
        for i, ini in zip(range(4),range(0, data.shape[0], fs*intervalo)):
            dados = pd.Series(data[ini:(ini+fs*2)])
            source = file.split('.')[0]
            character_class = file[i]
            df = df.append({'dados': dados, 'source_file': source, 'class': character_class}, ignore_index = True)
    return df

def main():
    train_set = create_set(train_path)    
    val_set = create_set(val_path)
    print(val_set['dados'])


if __name__ == '__main__':
    main()