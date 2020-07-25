import numpy as np
import os
from sklearn.model_selection import train_test_split

# Implementação de métodos funções genéricas
import commons as commons

# Implementação da extração das características
import extraction

# Implementação dos métodos de classificação
import svm as svm
import dtree as dtree
import multilayer_perceptron as mlpt

# Variáveis globais
CHARACTERISTICS_CSV = 'characteristics.csv'
CLASSIFICATION_CSV = 'classification.csv'
METHODS = [svm, dtree, mlpt]
TESTS = 100

def main(ofile, sep=';'):
    # Monta a base de dados das imagens
    db = commons.buildDatabase()

    # Carrega as características da imagem e salva na lista table
    print('[LOG] Extraindo as características')
    table = []
    if os.path.exists(CHARACTERISTICS_CSV):
        table = commons.loadCaracteristics(CHARACTERISTICS_CSV)
    else:
        table = extraction.extract(db)
        commons.saveCaracteristics(table, CHARACTERISTICS_CSV)

    # Define um vetor com os rótulos das imagens
    labelImage = np.array(['benigno']*20 + ['maligno']*20)

    # Abre o arquivo CSV de classificação para escrita
    with open(ofile, "w") as f:
        # Escreve o cabeçalho do arquivo
        header = ['id', 'channel', 'test_size'] + [m.NAME for m in METHODS]
        f.write(sep.join(header) + '\n')
        
        # Para cada canal
        idExperiment = 1
        for ch in ['red', 'green', 'blue', 'gray']:
            print(f'[LOG] Aplicando os métodos de classificação para o canal {ch}')

            # Para cada tamanho de grupo de teste
            for test_size in np.arange(0.1, 1.0, 0.1):
                # Inicia a lista que salva a acurácia do método para calcular a média
                for method in METHODS:
                    method.accur = []

                # Realiza consecutivos experimentos, visando calcular a média
                for experiment in range(TESTS):
                    while True:
                        # Separa em conjunto de treino/teste de acordo com o tamanho do grupo de teste definido
                        x_train, x_test, y_train, y_test = train_test_split(table[ch], labelImage, test_size=test_size, random_state=None)
                        
                        # O conjunto de teste sempre deve possuir pelo menos dois tipos (benigno/maligno)
                        if len(np.unique(y_train)) > 1:
                            break

                    # Para cada método, realizar a classificação com os conjuntos definidos e salva a acurácia
                    for method in METHODS:
                        method.accur.append(method.classify(x_train, x_test, y_train, y_test))

                # Escreve no arquivo a acurácia média do método
                f.write(sep.join([f'{idExperiment}', ch, f'{test_size:.02f}'] + [f'{np.mean(m.accur):.03f}' for m in METHODS]) + '\n')
                idExperiment += 1

if __name__ == "__main__":
    print('[LOG] A aplicação foi iniciada')

    main(CLASSIFICATION_CSV)

    print('[LOG] A aplicação foi encerrada')
    