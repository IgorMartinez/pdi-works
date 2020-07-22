def buildDatabase():
    return [f'{data}/{data} ({i}).tif' for data in ['beningno', 'maligno'] for i in range(1, 21)]

def loadCaracteristics(fname):
    table = {
        "red": [],
        "green": [],
        "blue": [],
        "gray": []
    }
    with open(fname, "w") as f:
        for line in f.readlines()[1:]:
            ch = line[0]
            table[ch].append(line[1:])
    table