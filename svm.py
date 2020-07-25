from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 

NAME = 'svm'

def classify(x_train, x_test, y_train, y_test):
    
    # Classifica as imagens com Support Vector Classifier (SVN)
    clf = SVC()

    # Ajusta o classificador
    clf.fit(x_train, y_train)

    # Faz a predição de cada imagem
    prediction = clf.predict(x_test)

    # Mede a acurácia para saber a performance do algoritmo
    acurracy = accuracy_score(prediction, y_test)

    return acurracy
