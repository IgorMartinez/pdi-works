from cv2 import cv2
import numpy as np
import math as math
import commons as cms

# Calcula o histograma normalizado da imagem
def normalizedHistogram(image):
    return np.array([np.count_nonzero(image == x) for x in range(256)]) / (image.shape[0] * image.shape[1])

# Calcula a média dos pixels da imagem
def mean(hist):
    m = 0
    for i in range(0, 256):
        m += hist[i] * i
    return m

# Calcula o momento n
def moment(n, hist, mean):
    aux = 0
    for i in range(256):
        aux += ((i-mean) ** n) * hist[i]
    return aux

# Calcula o desvio padrão da imagem
def deviation(hist, mean):
    return math.sqrt(moment(2, hist, mean))

# Calcula o terceiro momento da imagem
def third_moment(hist, mean):
    return moment(3, hist, mean)

# Calcula o quarto momento da imagem
def fourth_moment(hist, mean):
    return moment(4, hist, mean)

# Calcula a uniformidade da imagem
def uniformity(hist):
    unif = 0
    for i in range(0, 256):
        unif += hist[i] ** 2
    return unif

# Calcula a entropia da imagem
def entropy(hist):
    entr = 0
    for i in range(0, 256):
        entr += hist[i] * math.log(hist[i], 2) if hist[i] > 0 else 0
    return -entr

# Extrai as características de um canal da imagem 
def extractChannel(channel):
    histogram = normalizedHistogram(channel)
    avg = mean(histogram)
    std = deviation(histogram, avg)
    tdm = third_moment(histogram, avg)
    uni = uniformity(histogram)
    ent = entropy(histogram)
    ftm = fourth_moment(histogram, avg)
    return [avg, std, tdm, uni, ent, ftm]

# Realiza a extração de características de cada canal das imagens presentes na base de dados
def extract(database):
    # Especificação da matriz de cada canal
    #
    #       | média | desvio padrão | 3 momento | uniformidade | entropia | 4 momento |
    # ------+-------+---------------+-----------+--------------+----------+-----------|
    # img01 |  ...  |      ...      |    ...    |     ...      |    ...   |     ...   |
    # img02 |  ...  |      ...      |    ...    |     ...      |    ...   |     ...   |
    #  ...  |  ...  |      ...      |    ...    |     ...      |    ...   |     ...   |
    # img40 |  ...  |      ...      |    ...    |     ...      |    ...   |     ...   |
    #
    # Define as tabelas que serão armazenadas as estatísitica de cada canal
    table = {
        "red" : [],
        "green" : [],
        "blue": [],
        "gray": []
    }
    # Carrega cadaa imagem da base de dados e extrai suas características
    # para cada canal
    for fname in database:
        image = cms.loadImage(fname)
        # Para cada canal da imagem, estraí suas características
        # e as insere na tabela
        for ch in ["red", "green", "blue", "gray"]:
            channel = image[ch]
            chars = extractChannel(channel)
            table[ch].append(chars)
    # Retorna a tabela
    return table