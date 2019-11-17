from util import extract_attributes, getAudioSegments, getFilenames, loadAudio
from config import PATH_TREINAMENTO, PATH_VALIDACAO, AUDIOS_POR_ARQUIVO, INTERVALO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import librosa
import librosa.display

def plotAudio(audio, fs):
    librosa.display.waveplot(audio, sr = fs)
    plt.show()    

def trimAudio(audio):
    return librosa.effects.trim(y = audio, top_db = 30)[0]

def adjustAudio(audio, fs):
    factor = len(audio)/fs
    factor = factor/(INTERVALO*AUDIOS_POR_ARQUIVO)
    
    return librosa.effects.time_stretch(y = audio, rate = factor)

path = PATH_TREINAMENTO
files = getFilenames(PATH_TREINAMENTO)

filename = files[8]

fullAudio, fs = loadAudio(path+filename)
#fullAudio = adjustAudio(fullAudio, fs)
audio_segs, _ = getAudioSegments(path+filename)
'''
for i, audio in enumerate(audio_segs.values()):
    audio_segs[i] = trimAudio(audio_segs[i])
'''
print(f"Arquivo: {filename}, {fs}")

'''
plt.plot(fullAudio)
plt.show()
'''

#plotAudio(fullAudio, fs)
#print(len(fullAudio))

plotAudio(audio_segs[0], fs)
plotAudio(audio_segs[1], fs)
plotAudio(audio_segs[2], fs)
plotAudio(audio_segs[3], fs)

'''
for audio in audio_segs:
    plotAudio(audio)
'''
