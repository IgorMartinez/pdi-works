from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

METHOD_NAME = 'dtree'

def classify(x_train, x_test, y_train, y_test):
    print('[LOG] Aplicando o algoritmo de classificação Decision Tree')
    
    # Instancia um objeto da classe do classificador
    dtc = tree.DecisionTreeClassifier()

    # Ajusta o modelo
    dtc = dtc.fit(x_train, y_train)

    # Faz a predição de cada imagem
    dtc_predicton = dtc.predict(x_test)

    # Mede a acurácia para saber a performance do algoritmo
    acurracy = accuracy_score(dtc_predicton, y_test)
    print(f'[LOG] A acurácia do algoritmo foi de {acurracy}')

    return acurracy