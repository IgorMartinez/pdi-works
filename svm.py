from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 

def support_vector(x_train, x_test, y_train, y_test):
    print('[LOG] Aplicando o algoritmo de classificação Support Vector')

    # Classifica as imagens com Support Vector Classifier (SVN)
    clf = SVC()

    # Ajusta o classificador
    clf.fit(x_train, y_train)

    # Faz a predição de cada imagem
    prediction = clf.predict(x_test)

    # Mede a acurácia para saber a performance do algoritmo
    acurracy = accuracy_score(prediction, y_test)
    print(f'[LOG] A acurácia do algoritmo foi de {acurracy}')

    return acurracy