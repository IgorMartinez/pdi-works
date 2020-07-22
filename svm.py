#importações para classificação
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def support_vector(x_train, x_test, y_train, y_test):
    print('[LOG] Aplicando o algoritmo de classificação Support Vector')

    # Classifica as imagens com Support Vector Classifier (SVN)
    SVC_model = SVC()

    # Ajusta o classificador
    SVC_model.fit(x_train, y_train)

    # Faz a predição de cada imagem
    SVC_prediction = SVC_model.predict(x_test)

    # Mede a acurácia para saber a performance do algoritmo
    acurracy = accuracy_score(SVC_prediction, y_test)
    print(f'[LOG] A acurácia do algoritmo foi de {acurracy}')

    return acurracy