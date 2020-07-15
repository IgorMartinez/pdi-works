from cv2 import cv2
import numpy as np
import math as math
from extract import *
from classification import *

if __name__ == '__main__':
    print('[LOG] O programa foi iniciado')

    # Monta a base de dados
    database = []
    vetor_y = ['benigno'] * 20 + ['maligno'] * 20
    for data in ['benigno', 'maligno']:
        for i in range(1, 21):
            database.append(f'{data}/{data} ({i}).tif')

    # Extrai as características
    # tables = {"red": [], "green": [], "blue": [], "gray": []}
    # [media, desvio padrao, 3 momento, uniformidade, entropia, 4 momento]
    tables = extraction_characteristic(database)

    # Classifica
    for ch in ["red", "green", "blue", "gray"]:
        print(f'[LOG] Acurácia canal {ch}')
        extraction_classification(tables[ch], vetor_y)

    print('[LOG] O programa será encerrado')
