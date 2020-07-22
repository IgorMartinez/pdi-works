#importações para classificação
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def extraction_classification(X, y):
    # Test_size diz a porcentagem dos dados que usaremos para teste
    # Random_state é um parâmetro para pegar os dados de forma aleatória
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None)
    #print(X_train)
    #print(y_train)

    # Classifica as imagens com Support Vector Classifier (SVN)
    SVC_model = SVC()

    # Ajusta o classificador
    SVC_model.fit(X_train, y_train)

    # Faz a predição de cada imagem
    SVC_prediction = SVC_model.predict(X_test)

    # Mede a acurácia para saber a performance do algoritmo
    print(accuracy_score(SVC_prediction, y_test))

    # Cria a matriz de classificação, a qual apresenta mais características da performance
    print(confusion_matrix(SVC_prediction, y_test))