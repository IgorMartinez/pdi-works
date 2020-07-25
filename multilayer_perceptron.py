from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import warnings

# Ignora todos os Warnings gerados
warnings.filterwarnings('ignore')

NAME = 'multilayer_perceptron'

def classify(x_train, x_test, y_train, y_test):

    # Instancia um objeto da classe do classificador
    clf = MLPClassifier(solver='lbfgs')

    # Escala o conjunto de treino
    x_scaled = preprocessing.scale(x_train)
    
    # Ajusta o modelo
    clf.fit(x_scaled, y_train)

    # Faz a predição de cada imagem
    prediction = clf.predict(x_test)

    # Mede a acurácia para saber a performance do algoritmo
    acurracy = accuracy_score(prediction, y_test)

    return acurracy