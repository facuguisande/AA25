from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, fbeta_score
from arboles_decision import ArbolDecisionPersonalizado
import numpy as np

def entrenar_y_evaluar_arbol(X_train, y_train, X_test, y_test, min_gain_info=0.0, max_profundidad=None):
    # Entrenar el árbol con  hiperparámetros
    arbol = ArbolDecisionPersonalizado(min_gain_info=min_gain_info, max_profundidad=max_profundidad)
    arbol.fit(X_train, y_train)

    # Predecir
    y_pred = arbol.predecir(X_test)

    # Calcular métricas de evaluación
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = fbeta_score(y_test, y_pred, beta=2, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, f1, cm, y_pred