from sklearn import tree
from sklearn.metrics import accuracy_score

def decision_tree(x_train, x_test, y_train, y_test):
    print('[LOG] Aplicando o algoritmo de classificação Decision Tree')
    
    # Instancia um objeto da classe do classificador
    clf = tree.DecisionTreeClassifier()

    # Ajusta o modelo
    clf.fit(x_train, y_train)

    # Faz a predição de cada imagem
    prediction = clf.predict(x_test)

    # Mede a acurácia para saber a performance do algoritmo
    acurracy = accuracy_score(prediction, y_test)
    print(f'[LOG] A acurácia do algoritmo foi de {acurracy}')

    return acurracy