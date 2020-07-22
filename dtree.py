from sklearn import tree
from sklearn.metrics import accuracy_score

NAME = 'decision_tree'

def classify(x_train, x_test, y_train, y_test):
    # Instancia um objeto da classe do classificador
    clf = tree.DecisionTreeClassifier()

    # Ajusta o modelo
    clf.fit(x_train, y_train)

    # Faz a predição de cada imagem
    prediction = clf.predict(x_test)

    # Mede a acurácia para saber a performance do algoritmo
    acurracy = accuracy_score(prediction, y_test)

    return acurracy