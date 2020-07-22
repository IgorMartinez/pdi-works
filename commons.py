import cv2
import numpy as np

# Constrói um vetor com todos os nomes dos arquivos das imagens analizadas
def buildDatabase():
    return ['database/%s/%s (%s).tif' % (data,data,i) for data in ['benigno', 'maligno'] for i in range(1, 21)]
# Carrega a tabela de características a partir de um arquivo CSV
def loadCaracteristics(fname, sep=';'):
    table = {
        "red": [],
        "green": [],
        "blue": [],
        "gray": []
    }
    with open(fname, "r") as f:
        for line in f.readlines()[1:]:
            values = line.split(sep)
            ch = values[2]
            table[ch].append(values[3:])
    return table

# Carrega as imagens da base de dados e separa por canal de cor
def loadImage(fname):
    img = cv2.imread(fname)
    blue, green, red = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "red": red, 
        "green": green,
        "blue": blue,
        "gray": gray
    }
    
# Salva a tabela de características, por canal, em um arquivo CSV
def saveCaracteristics(table, ofname, sep=';'):
    header = ["id", "file", "channel", "mean", "deviation", "3rd", "uniformity", "entropy", "4th"]
    db = buildDatabase()
    xp = 1
    with open(ofname, "w") as f:
        # Prints the CSV header
        f.write(sep.join(header) + '\n')
        for channel in table.keys():
            for i in range(len(table[channel])):
                f.write(sep.join([str(xp), db[i], channel] + [str(x) for x in table[channel][i]]) + '\n')
                xp += 1