from cv2 import cv2
import numpy as np
import math as math
from extract import *
from svm import *
from commons import *

if __name__ == '__main__':
    print('[LOG] O programa foi iniciado')

    # Monta a base de dados
    database = buildDatabase()
    vetor_y = ['benigno'] * 20 + ['maligno'] * 20

    # Extrai as características
    # tables = {"red": [], "green": [], "blue": [], "gray": []}
    # [media, desvio padrao, 3 momento, uniformidade, entropia, 4 momento]
    tables = extraction_characteristic(database)

    # Classifica
    for ch in ["red", "green", "blue", "gray"]:
        print(f'[LOG] Acurácia canal {ch}')
        extraction_classification(tables[ch], vetor_y)

    print('[LOG] O programa será encerrado')

if __name__ == "__main__":
    y = np.array(['benigno']*20 + ['maligno']*20)
    db = buildDatabase()
    if 