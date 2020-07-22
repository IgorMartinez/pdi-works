import cv2

def buildDatabase():
    return [f'{data}/{data} ({i}).tif' for data in ['beningno', 'maligno'] for i in range(1, 21)]

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

def saveCaracteristics(rgbTable, ofname, sep=';'):
    header = ["id", "file", "channel", "mean", "deviation", "3rd", "uniformity", "entropy", "4th"]
    db = buildDatabase()
    xp = 1
    with open(ofname, "w") as f:
        # Prints the CSV header
        f.write(sep.join(header) + '\n')
        for chan in rgbTable.keys():
            for i in range(len(rgbTable[chan])):
                f.write(sep.join([str(xp), db[i], chan] + [str(x) for x in rgbTable[chan][i]]) + '\n')
                xp += 1