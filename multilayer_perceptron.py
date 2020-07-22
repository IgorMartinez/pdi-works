from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

NAME = 'multilayer_perceptron'

def classify(x_train, x_test, y_train, y_test):

    
    # Instancia um objeto da classe do classificador
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    # Ajusta o modelo
    clf.fit(x_train, y_train)

    # Faz a predição de cada imagem
    prediction = clf.predict(x_test)

    # Mede a acurácia para saber a performance do algoritmo
    acurracy = accuracy_score(prediction, y_test)


    return acurracy