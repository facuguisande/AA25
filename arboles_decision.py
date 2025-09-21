import numpy as np
from sklearn.metrics import precision_score
from collections import Counter

class NodoArbol:
    def __init__(self, indice_atributo=None, valor_division=None, valor_prediccion=None, ganancia=None, 
                 rama_izquierda=None, rama_derecha=None):
        self.indice_atributo = indice_atributo      # Índice del atributo para dividir
        self.valor_division = valor_division        # Valor de corte para la división
        self.valor_prediccion = valor_prediccion    # Valor de predicción si es hoja
        self.ganancia = ganancia                    # Ganancia de información en la división
        self.rama_izquierda = rama_izquierda       
        self.rama_derecha = rama_derecha            

class ArbolDecisionPersonalizado:
    def __init__(self, min_gain_info=0.01, max_profundidad=20, min_muestras_division=1):
        self.min_gain_info = min_gain_info
        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        self.raiz = None
        
    def ajustar_pesos(self, y):
        # Ajusta los pesos de las clases para manejar desequilibrios
        contador = Counter(y)
        n_muestras = len(y)
        clases_unicas = np.unique(y)
        # Cálculo similar a sklearn: n_samples / (n_classes * np.bincount(y)) , para distribuir los pesos
        self.pesos_clases = {cls: n_muestras / (len(clases_unicas) * contador[cls]) 
                            for cls in clases_unicas}
    def fit(self, X, y):
        # Entrena nuestro arbol
        self.ajustar_pesos(y)
        self.raiz = self.construir_arbol(X, y, profundidad=0)
        return self
    
    def entropia(self, y):
        clases = np.unique(y)
        if len(clases) <= 1:
            return 0
        
        # Aplica los pesos de clase al calcular las proporciones
        contador = Counter(y)
        n_muestras = sum(contador[cls] * self.pesos_clases[cls] for cls in contador)
        proporciones = np.array([contador[cls] * self.pesos_clases[cls] / n_muestras for cls in clases])
        
        # Evitar log(0)
        proporciones = proporciones[proporciones > 0]
        return -np.sum(proporciones * np.log2(proporciones))
    
    def ganancia_informacion(self, y, y_izquierda, y_derecha):
        # Aplicamos pesos de clase en el cálculo de la ganancia de información
        # Pesos ponderados para cada subconjunto
        peso_izq = sum(self.pesos_clases[cls] for cls in y_izquierda)
        peso_der = sum(self.pesos_clases[cls] for cls in y_derecha)
        peso_total = peso_izq + peso_der
        
        # Ganancia = entropia padre - suma ponderada de entropias de los hijos
        ganancia = self.entropia(y) - (
            (peso_izq / peso_total) * self.entropia(y_izquierda) + 
            (peso_der / peso_total) * self.entropia(y_derecha)
        )
        return ganancia
    
    def mejor_division(self, X, y):
        # Busca la mejor división posible en los atributos, dado su gini
        n_muestras, n_atributos = X.shape
        
        if n_muestras <= self.min_muestras_division:
            return None, None, None
        
        mejor_ganancia = -1
        mejor_atributo = None
        mejor_division = None

        for idx_atributo in range(n_atributos):
            valores_unicos = np.unique(X[:, idx_atributo])

            for val_div in valores_unicos:
                # División de datos
                indices_izq = X[:, idx_atributo] == val_div
                indices_der = ~indices_izq
                
                if np.sum(indices_izq) == 0 or np.sum(indices_der) == 0:
                    continue
                
                y_izquierda, y_derecha = y[indices_izq], y[indices_der]
                ganancia = self.ganancia_informacion(y, y_izquierda, y_derecha)
                
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_atributo = idx_atributo
                    mejor_division = val_div

        return mejor_atributo, mejor_division, mejor_ganancia


    def construir_arbol(self, X, y, profundidad):
        # Construimos el arbol de manera recursiva, unsando los criterios de parada
        n_muestras, n_atributos = X.shape
        n_clases = len(np.unique(y))
        
        # Criterios de parada
        if (profundidad >= self.max_profundidad or 
            n_muestras < self.min_muestras_division or 
            n_clases == 1):
            # Crear nodo hoja
            contador = Counter(y)
            # Ajustar por pesos de clase para la predicción final
            mejor_clase = max(contador.items(), 
                             key=lambda x: x[1] * self.pesos_clases[x[0]])[0]
            return NodoArbol(valor_prediccion=mejor_clase)
        
        # Buscar la mejor división
        atributo_idx, val_div, ganancia = self.mejor_division(X, y)
        
        # Si no hay división con suficiente ganancia, crear nodo hoja
        if atributo_idx is None or ganancia < self.min_gain_info:
            contador = Counter(y)
            mejor_clase = max(contador.items(), 
                             key=lambda x: x[1] * self.pesos_clases[x[0]])[0]
            return NodoArbol(valor_prediccion=mejor_clase)

        # Dividir los datos
        indices_izq = X[:, atributo_idx] == val_div
        indices_der = ~indices_izq
        
        # Construir subárboles
        subárbol_izq = self.construir_arbol(
            X[indices_izq], y[indices_izq], profundidad + 1)
        subárbol_der = self.construir_arbol(
            X[indices_der], y[indices_der], profundidad + 1)
        
        return NodoArbol(
            indice_atributo=atributo_idx,
            valor_division=val_div,
            ganancia=ganancia,
            rama_izquierda=subárbol_izq,      
            rama_derecha=subárbol_der         
        )
    
    # Ingres matriz con dataset
    # Salida: vector con predicciones
    def predecir(self, X):
        return np.array([self.predecir_muestra(x, self.raiz) for x in X])
    
    def predecir_muestra(self, x, nodo):
        # Si es un nodo hoja, devolver el valor
        if nodo.valor_prediccion is not None:
            return nodo.valor_prediccion

        # Decidir qué camino seguir, esto sigue hasta una hoja
        if x[nodo.indice_atributo] == nodo.valor_division:
            return self.predecir_muestra(x, nodo.rama_izquierda)
        else:
            return self.predecir_muestra(x, nodo.rama_derecha)
