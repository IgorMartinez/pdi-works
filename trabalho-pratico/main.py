from cv2 import cv2
import numpy as np
import math as math

def extraction_characteristic(database):
    # Posição no vetor
    # [média, desvio padrão, 3 momento, uniformidade, entropia, 4 momento]

    # Define as tabelas que serão armazenadas
    tables = [[], [], [], []]

    for data in database:
        # Lê a imagem colorida
        image = cv2.imread(f'database/{data}')

        print(f'[LOG] Extraindo as características da imagem {data}')

        # Recebe a imagem em cinza
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Recebe largura, altura e canais da imagem
        width, height, _ = image.shape
        number_pixels = width * height

        # Linhas da tabela
        line = [[], [], [], []]

        # Conta quantas vezes cada intensidade de cor apareceu na imagem
        hist = [np.zeros(256), np.zeros(256), np.zeros(256), np.zeros(256)]
        for i in range(0, width):
            for j in range(0, height):
                hist[0][image[i, j, 0]] += 1
                hist[1][image[i, j, 1]] += 1
                hist[2][image[i, j, 2]] += 1
                hist[3][image_gray[i, j]] += 1

        # Normaliza e calcula as médias
        for h in range(0, 4):
            line[h].append(0)
            for i in range(0, 256):
                # Normaliza os histogramas
                hist[h][i] /= number_pixels

                # Calcula as médias
                line[h][0] += hist[h][i] * i

        # Desvio padrão, Terceiro momento, uniformidade, entropia
        for h in range(0, 4):
            line[h].extend([0, 0, 0, 0, 0])
            for i in range(0, 256):
                # Calcula desvio padrão (necessário calcular raiz)
                line[h][1] += ((i-line[h][0]) ** 2) * hist[h][i]

                # Calcula o terceiro momento
                line[h][2] += ((i-line[h][0]) ** 3) * hist[h][i]

                # Calcula uniformidade
                line[h][3] += hist[h][i] ** 2

                # Calcula entropia (necessário inverter valor)
                line[h][4] += hist[h][i] * math.log(hist[h][i], 2) if hist[h][i] > 0 else 0

                # Calcula o quarto momento
                line[h][5] += ((i-line[h][0]) ** 4) * hist[h][i]

            # Finaliza cálculo do desvio padrão
            line[h][1] = math.sqrt(line[h][1])

            # Finaliza cálculo da entropia
            line[h][4] *= -1

        # Adiciona nas tabelas os dados da linha
        for h in range(0, 4):
            tables[h].append(line[h])
        
    return tables

if __name__ == '__main__':
    print('[LOG] O programa foi iniciado')

    # Monta a base de dados
    database = []
    for data in ['benigno', 'maligno']:
        for i in range(1, 21):
            database.append(f'{data}/{data} ({i}).tif')

    # Extrai as características
    tables = extraction_characteristic(database)

    print(tables[0])

    print(tables[0][0])

    print('[LOG] O programa será encerrado')
