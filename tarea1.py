import pandas as pd
import numpy as np
from collections import Counter
from functools import reduce
import sklearn as sk
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, fbeta_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB

from imblearn.over_sampling import RandomOverSampler

from astral import moon


## Clases y funciones para el árbol de decisión
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


## Clases y funciones para Naive Bayes
# ### Calculo de probabilidades para construir modelo
def calcular_probabilidad(df, col, valor, clase, m):
    # Filtrar el DataFrame por la clase
    df_clase = df[df['manana_llueve'] == clase]
    cant_elementos = len(df_clase)
    p=1/len(df[col].unique())
    # Calcular la probabilidad
    if(cant_elementos == 0):
        print("No hay elementos de la clase ", clase, " para calcular la probabilidad")
        return 0
    #Utilizamos m-estimador (e + m * p) / (n + m)
    return ((df_clase[col].value_counts().get(valor, 0) + m *p) / (cant_elementos + m))


# ### Predicción de clase con productoria de probabilidades
def predecir_clase(fila, modelo, prob_iniciales):
    prob_clases = [1, 1]
    for i in range(2):
        for col in fila.index:
            if col != 'manana_llueve':
                valor = fila[col]
                # Buscar la probabilidad correspondiente en el modelo
                prob = None
                for item in modelo[i]:
                    if item[0] == col and item[1] == valor:
                        prob = item[2]
                        break
                prob_clases[i] *= prob
        # Multiplicamos por la probabilidad de la clase
        prob_clases[i] *= prob_iniciales[i]
    # Devolvemos la clase con mayor probabilidad
    if(prob_clases[0]>prob_clases[1]):
        return 0
    else:
        return 1

# ### Predicción de clase con logaritmo
def predecir_clase_log(fila, modelo, prob_iniciales):
    import math
    
    # Usamos logaritmos para evitar underflow numérico
    log_prob_clases = [0, 0]  # Inicializamos en 0 (log(1) = 0)
    
    for i in range(2):
        # Comenzamos con el logaritmo de la probabilidad inicial de la clase
        log_prob_clases[i] = math.log(prob_iniciales[i])
        
        for col in fila.index:
            if col != 'manana_llueve':
                valor = fila[col]
                # Buscar la probabilidad correspondiente en el modelo
                prob = None
                for item in modelo[i]:
                    if item[0] == col and item[1] == valor:
                        prob = item[2]
                        break
                
                # Si encontramos la probabilidad, sumamos su logaritmo
                if prob is not None and prob > 0:
                    log_prob_clases[i] += math.log(prob)
                else:
                    # Si prob es 0 o None, asignamos un valor muy pequeño para evitar log(0)
                    log_prob_clases[i] += math.log(1e-10)
    
    # Devolvemos la clase con mayor log-probabilidad
    if log_prob_clases[0] > log_prob_clases[1]:
        return 0
    else:
        return 1

# ### Codificar atributos
#Solo trasnforma la categorias a numeros sin perder la relacion entre ellas, para usarse en modelo que no acepta texto
def encode(str):
    if str == 'Baja' or str == 'N' or str == "Verano" or str == 'Luna Nueva':
        return 0
    elif str == 'Media' or str == 'NE' or str == 'Otoño' or str == 'Cuarto Creciente':
        return 1
    elif str == 'Alta' or str == 'E' or str == 'Invierno' or str == 'Luna Llena':
        return 2
    elif str == 'SE' or str == 'Primavera' or str == 'Cuarto Menguante':
        return 3
    elif str == 'S':
        return 4
    elif str == 'SW':
        return 5
    elif str == 'W':
        return 6
    elif str == 'NW':
        return 7
    elif str == 0 or str == 1:
        return int(str)


# ### Clase (objeto) que representa al modelo
class NaiveBayes:
    def __init__(self, m):
        self.m = m
        self.modelo = [[], []]  # Lista para almacenar las probabilidades de cada clase
        self.prob_iniciales = [0, 0]  # Probabilidades iniciales de cada clase

    def fit(self, X, y):
        df=X.copy()
        df['manana_llueve']=y
        # Calcular las probabilidades iniciales de cada clase
        total_filas = len(df)
        self.prob_iniciales[0] = len(df[df['manana_llueve'] == 0]) / total_filas
        self.prob_iniciales[1] = 1-self.prob_iniciales[0]
        #Para cada clase, para cada columna, para cada valor posible, calcular la probabilidad y almacenarla
        for i in range(2):
            for col in df.columns:
                if col != 'fecha' and col != 'manana_llueve':
                    # Columnas categóricas
                    if df[col].dtype.name == 'category':
                        valores_posibles = df[col].cat.categories.tolist()
                    else:
                        # Columnas numéricas
                        valores_posibles = df[col].unique().tolist()
                    for valor in valores_posibles:
                        prob = calcular_probabilidad(df, col, valor, i, self.m)
                        self.modelo[i].append((col, valor, prob))

    def predict(self, X):
        # Predecir la clase para cada fila en X
        return X.apply(lambda fila: predecir_clase(fila, self.modelo, self.prob_iniciales), axis=1)


## Función principal
def main():
    CSV_PATH = './Dataset_INUMET/CSV/'
    CSV_PATH = input("Ingrese el path donde se encuentran los archivos CSV (debe terminar con /): ").strip()

    # Definimos los archivos CSV
    archivo_lluvia = CSV_PATH + 'Lluvia.csv'
    archivo_temperatura = CSV_PATH + 'Temperatura.csv'
    archivo_humedad = CSV_PATH + 'Humedad.csv'
    archivo_viento = CSV_PATH + 'VientoCarrasco.csv'
    archivo_fenomenos = CSV_PATH + 'Fenomenos.csv'

    # Los cargamos en dataframes
    try:
        lluvia_df = pd.read_csv(archivo_lluvia, sep=';')
        temperatura_df = pd.read_csv(archivo_temperatura, sep=';')
        humedad_df = pd.read_csv(archivo_humedad, sep=';')
        viento_df = pd.read_csv(archivo_viento, sep=';')
        fenomenos_df = pd.read_csv(archivo_fenomenos, sep=';')
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        exit()

    ########################
    ### Preprocesamiento ###
    ########################
    print('\nPreprocesamiento')
    print('Iniciando el preprocesamiento de los datos...')
    # Definimos el rango a partir de los datos y nuestras consideraciones (en el informe se explica)
    rango = (pd.to_datetime(humedad_df.iloc[0, 0], format='%d/%m/%Y'), pd.to_datetime(lluvia_df.iloc[lluvia_df.last_valid_index() - 1, 2], format='%d/%m/%Y'))

    ## Lluvias
    # Nos quedamos con las filas dentro del rango definido
    lluvia_df_subset = lluvia_df[(pd.to_datetime(lluvia_df.iloc[:, 2], format='%d/%m/%Y') >= rango[0])].copy()
    lluvia_df_subset.reset_index(drop=True, inplace=True)
    # Nos quedamos solamente con las columnas de la fecha y los mm de lluvia
    lluvia_df_subset = lluvia_df_subset.iloc[:, [2, 3]]
    lluvia_df_subset.columns = ['fecha', 'mm_lluvia']
    # Normalizamos el formato de la fecha (pasamos a pandas datetime64[ns])
    lluvia_df_subset['fecha'] = pd.to_datetime(lluvia_df_subset['fecha'], format='%d/%m/%Y')
    # Verificamos si hay valores no numéricos en lluvia y los reemplazamos por 0, por lo que comentó el profe. Esto se puede hacer para todo el dataset porque reemplazamos por valor constante.
    lluvia_df_subset['mm_lluvia'] = pd.to_numeric(lluvia_df_subset['mm_lluvia'], errors='coerce').fillna(0)
    # Definimos día lluvioso como los días con lluvias mayores a 1mm
    umbral_lluvia = 1.0
    # Creamos la columna "día lluvioso" (nd = 1)
    lluvia_df_subset['dia_lluvioso'] = (lluvia_df_subset['mm_lluvia'] >= umbral_lluvia).astype(int)
    # Creamos la columna "semana lluviosa" (nd = 7), que indica la cantidad de días lluviosos en los últimos 7 días (incluyendo el día actual)
    lluvia_df_subset['semana_lluviosa'] = lluvia_df_subset['dia_lluvioso'].rolling(window=7, min_periods=1).sum().astype(int)
    # Creamos la columna "quincena lluviosa" (nd = 15), que indica la cantidad de días lluviosos en los últimos 15 días (incluyendo el día actual)
    lluvia_df_subset['quincena_lluviosa'] = lluvia_df_subset['dia_lluvioso'].rolling(window=15, min_periods=1).sum().astype(int)
    # Creamos la columna "mes lluvioso" (nd = 30), que indica la cantidad de días lluviosos en los últimos 30 días (incluyendo el día actual)
    lluvia_df_subset['mes_lluvioso'] = lluvia_df_subset['dia_lluvioso'].rolling(window=30, min_periods=1).sum().astype(int)
    # Creamos la clase objetivo "mañana llueve"
    lluvia_df_subset['manana_llueve'] = lluvia_df_subset['dia_lluvioso'].shift(-1)
    # Eliminamos la última fila que ahora tiene un valor NaN en 'manana_llueve'
    lluvia_df_subset = lluvia_df_subset.dropna(subset=['manana_llueve'])
    lluvia_df_subset['manana_llueve'] = lluvia_df_subset['manana_llueve'].astype(int)

    ## Temperatura
    # Nos quedamos con las filas dentro del rango definido
    temperatura_df_subset = temperatura_df[(pd.to_datetime(temperatura_df.iloc[:, 0], format='%d/%m/%Y') >= rango[0]) & (pd.to_datetime(temperatura_df.iloc[:, 0], format='%d/%m/%Y') <= rango[1])].copy()
    temperatura_df_subset.reset_index(drop=True, inplace=True)
    # Nos quedamos solamente con las columnas fecha, Tmax y Tmin de Carrasco
    temperatura_df_subset = temperatura_df_subset.iloc[:, :3]
    temperatura_df_subset.columns = ['fecha', 'temp_max', 'temp_min']
    # Normalizamos el formato de la fecha (pasamos a pandas datetime64[ns])
    temperatura_df_subset['fecha'] = pd.to_datetime(temperatura_df_subset['fecha'], format='%d/%m/%Y', errors='coerce')
    # Eliminamos filas con fechas inválidas
    temperatura_df_subset = temperatura_df_subset.dropna(subset=['fecha'])
    # Creamos columnas temp_max_semana, temp_min_semana con el promedio de la semana anterior (nd = 7)
    temperatura_df_subset['temp_max_semana'] = temperatura_df_subset['temp_max'].rolling(window=7, min_periods=1).mean()
    temperatura_df_subset['temp_min_semana'] = temperatura_df_subset['temp_min'].rolling(window=7, min_periods=1).mean()
    # Creamos columnas temp_max_quincena, temp_min_quincena con el promedio de los últimos 15 días (nd = 15)
    temperatura_df_subset['temp_max_quincena'] = temperatura_df_subset['temp_max'].rolling(window=15, min_periods=1).mean()
    temperatura_df_subset['temp_min_quincena'] = temperatura_df_subset['temp_min'].rolling(window=15, min_periods=1).mean()
    # Creamos columnas temp_max_mes, temp_min_mes con el promedio del último mes (nd = 30)
    temperatura_df_subset['temp_max_mes'] = temperatura_df_subset['temp_max'].rolling(window=30, min_periods=1).mean()
    temperatura_df_subset['temp_min_mes'] = temperatura_df_subset['temp_min'].rolling(window=30, min_periods=1).mean()
    # Para categorizar hay que considerar antes valores de todo el dataset. Para hacerlo correctamente primero debemos separar en conjunto de entrenamiento y prueba.
    # Se hace más adelante.

    ## Humedad
    # Nos quedamos con las filas dentro del rango definido
    humedad_df_subset = humedad_df[(pd.to_datetime(humedad_df.iloc[:, 0], format='%d/%m/%Y %H:%M', errors='coerce') >= rango[0]) & (pd.to_datetime(humedad_df.iloc[:, 0], format='%d/%m/%Y %H:%M', errors='coerce') <= rango[1])].copy()
    humedad_df_subset.reset_index(drop=True, inplace=True)
    # Nos quedamos solamente con las columnas de la fecha y la humedad relativa de Carrasco
    humedad_df_subset = humedad_df_subset.iloc[:,:2].copy()
    humedad_df_subset.columns = ['fecha', 'humedad_relativa']
    # Normalizamos el formato de la fecha (pasamos a pandas datetime64[ns]), forzando errores a NaT
    humedad_df_subset['fecha'] = pd.to_datetime(humedad_df_subset['fecha'], format='%d/%m/%Y %H:%M', errors='coerce')
    # Eliminamos filas con fechas no válidas
    humedad_df_subset = humedad_df_subset.dropna(subset=['fecha'])
    # Transformamos la fecha a solo fecha (sin hora) y lo devolvemos a datetime64[ns] para ser consistentes
    humedad_df_subset['fecha'] = humedad_df_subset['fecha'].dt.date
    humedad_df_subset['fecha'] = pd.to_datetime(humedad_df_subset['fecha'])
    # Agrupamos por dia y calculamos la humedad relativa promedio diaria
    humedad_df_subset = humedad_df_subset.groupby('fecha').agg({'humedad_relativa': 'mean'}).reset_index()
    # Creamos la columna humedad_relativa_semana con el promedio de la semana anterior (nd = 7)
    humedad_df_subset['humedad_relativa_semana'] = humedad_df_subset['humedad_relativa'].rolling(window=7, min_periods=1).mean()
    # Creamos la columna humedad_relativa_quincena con el promedio de los últimos 15 días (nd = 15)
    humedad_df_subset['humedad_relativa_quincena'] = humedad_df_subset['humedad_relativa'].rolling(window=15, min_periods=1).mean()
    # Creamos la columna humedad_relativa_mes con el promedio del último mes (nd = 30)
    humedad_df_subset['humedad_relativa_mes'] = humedad_df_subset['humedad_relativa'].rolling(window=30, min_periods=1).mean()
    # Para categorizar hay que considerar antes valores de todo el dataset. Para hacerlo correctamente primero debemos separar en conjunto de entrenamiento y prueba.
    # Se hace más adelante.

    ## Viento
    # Nos quedamos con las filas dentro del rango definido
    viento_df_subset = viento_df[(pd.to_datetime(viento_df.iloc[:, 0], format='%d/%m/%Y %H:%M', errors='coerce') >= rango[0]) & (pd.to_datetime(viento_df.iloc[:, 0], format='%d/%m/%Y %H:%M', errors='coerce') <= rango[1])].copy()
    viento_df_subset.reset_index(drop=True, inplace=True)
    # Nos quedamos solamente con las columnas de la fecha y la dirección del viento de Carrasco
    viento_df_subset = viento_df_subset.iloc[:, :2].copy()
    viento_df_subset.columns = ['fecha', 'direccion_viento']
    # Normalizamos el formato de la fecha (pasamos a pandas datetime64[ns]), forzando errores a NaT
    viento_df_subset['fecha'] = pd.to_datetime(viento_df_subset['fecha'], format='%d/%m/%Y %H:%M', errors='coerce')
    # Eliminamos filas con fechas no válidas
    viento_df_subset = viento_df_subset.dropna(subset=['fecha'])
    # Transformamos la fecha a solo fecha (sin hora) y lo devolvemos a datetime64[ns] para ser consistentes
    viento_df_subset['fecha'] = viento_df_subset['fecha'].dt.date
    viento_df_subset['fecha'] = pd.to_datetime(viento_df_subset['fecha'])
    # Creamos funcion para categorizar la dirección del viento de grados a una de las 8 direcciones
    def categorizar_direccion_viento(grados):
        if grados is None or pd.isna(grados):
            return np.nan
        try:
            grados = float(grados)
        except (ValueError, TypeError):
            return np.nan
        dirs = [0, 1, 2, 3, 4, 5, 6, 7]
        #dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        ix = round(grados / 45) % 8
        return dirs[ix] 
    # La aplicamos al dataframe
    viento_df_subset['direccion_viento'] = viento_df_subset['direccion_viento'].apply(categorizar_direccion_viento)
    # Una vez categorizado, agrupamos por dia y calculamos la dirección del viento predominante diaria
    viento_df_subset = viento_df_subset.groupby('fecha').agg({'direccion_viento': lambda x: x.mode()[0] if not x.mode().empty else np.nan}).reset_index()
    # Estos atributos no tienen mucho sentido por si mismos, pero para ser consistentes con el resto del dataset los incluimos.
    # Creamos la columna direccion_viento_semana con la moda de la semana anterior (nd = 7). Usamos groupby porque es categórica
    viento_df_subset['direccion_viento_semana'] = viento_df_subset['direccion_viento'].rolling(window=7, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    # Creamos la columna direccion_viento_quincena con la moda de los últimos 15 días (nd = 15)
    viento_df_subset['direccion_viento_quincena'] = viento_df_subset['direccion_viento'].rolling(window=15, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    # Creamos la columna direccion_viento_mes con la moda del último mes (nd = 30)
    viento_df_subset['direccion_viento_mes'] = viento_df_subset['direccion_viento'].rolling(window=30, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    # Pasamos las columnas de dirección del viento de [0, 1, 2, 3, 4, 5, 6, 7] a tipo category con ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    viento_df_subset['direccion_viento'] = viento_df_subset['direccion_viento'].map({0: 'N', 1: 'NE', 2: 'E', 3: 'SE', 4: 'S', 5: 'SW', 6: 'W', 7: 'NW'}).astype('category')
    viento_df_subset['direccion_viento_semana'] = viento_df_subset['direccion_viento_semana'].map({0: 'N', 1: 'NE', 2: 'E', 3: 'SE', 4: 'S', 5: 'SW', 6: 'W', 7: 'NW'}).astype('category')
    viento_df_subset['direccion_viento_quincena'] = viento_df_subset['direccion_viento_quincena'].map({0: 'N', 1: 'NE', 2: 'E', 3: 'SE', 4: 'S', 5: 'SW', 6: 'W', 7: 'NW'}).astype('category')
    viento_df_subset['direccion_viento_mes'] = viento_df_subset['direccion_viento_mes'].map({0: 'N', 1: 'NE', 2: 'E', 3: 'SE', 4: 'S', 5: 'SW', 6: 'W', 7: 'NW'}).astype('category')

    ## Fenómenos
    # Nos quedamos con las filas dentro del rango definido
    fenomenos_df_subset = fenomenos_df[(pd.to_datetime(fenomenos_df.iloc[:, 0], format='%d/%m/%Y', errors='coerce') >= rango[0]) & (pd.to_datetime(fenomenos_df.iloc[:, 0], format='%d/%m/%Y', errors='coerce') <= rango[1])].copy()
    fenomenos_df_subset.reset_index(drop=True, inplace=True)
    # Nos quedamos solamente con las columnas de la fecha y el fenómeno
    fenomenos_df_subset = fenomenos_df_subset.iloc[:, :2].copy()
    fenomenos_df_subset.columns = ['fecha', 'fenomeno']
    # Normalizamos el formato de la fecha (pasamos a pandas datetime64[ns])
    fenomenos_df_subset['fecha'] = pd.to_datetime(fenomenos_df_subset['fecha'], format='%d/%m/%Y', errors='coerce')
    # Eliminamos filas con fechas no válidas
    fenomenos_df_subset = fenomenos_df_subset.dropna(subset=['fecha'])
    # Transformamos la fecha a solo fecha (sin hora) y lo devolvemos a datetime64[ns] para ser consistentes
    fenomenos_df_subset['fecha'] = fenomenos_df_subset['fecha'].dt.date
    fenomenos_df_subset['fecha'] = pd.to_datetime(fenomenos_df_subset['fecha'])
    # Agrupamos por dia y marcamos si hubo tormenta con truenos (17) ese dia (1 si hubo, 0 si no)
    fenomenos_df_subset['hay_truenos'] = fenomenos_df_subset['fenomeno'].apply(lambda x: 1 if x == 17 else 0)
    fenomenos_df_subset = fenomenos_df_subset.groupby('fecha').agg({'hay_truenos': 'max'}).reset_index()
    # Creamos la columna hay_truenos_semana con la cantidad de días con truenos en la semana anterior (nd = 7)
    fenomenos_df_subset['hay_truenos_semana'] = fenomenos_df_subset['hay_truenos'].rolling(window=7, min_periods=1).sum().astype(int)
    # Creamos la columna hay_truenos_quincena con la cantidad de días con truenos en los últimos 15 días (nd = 15)
    fenomenos_df_subset['hay_truenos_quincena'] = fenomenos_df_subset['hay_truenos'].rolling(window=15, min_periods=1).sum().astype(int)
    # Creamos la columna hay_truenos_mes con la cantidad de días con truenos en el último mes (nd = 30)
    fenomenos_df_subset['hay_truenos_mes'] = fenomenos_df_subset['hay_truenos'].rolling(window=30, min_periods=1).sum().astype(int)

    ## Estación
    # Tomamos el dataframe con mayor cantidad de filas (lluvia) como base y nos quedamos con la columna fecha
    estacion_df = lluvia_df_subset[['fecha']].copy()
    # Creamos funcion para determinar la estación del año según la fecha
    def determinar_estacion(fecha):
        mes = fecha.month
        dia = fecha.day
        if (mes == 12 and dia >= 21) or (mes <= 3 and (mes != 3 or dia < 20)):
            return 0
            # return 'Verano'
        elif (mes == 3 and dia >= 20) or (mes <= 6 and (mes != 6 or dia < 21)):
            return 1
            # return 'Otoño'
        elif (mes == 6 and dia >= 21) or (mes <= 9 and (mes != 9 or dia < 22)):
            return 2
            # return 'Invierno'
        else:
            return 3
            # return 'Primavera'
    # Creamos la columna estación aplicando la función
    estacion_df['estacion'] = estacion_df['fecha'].apply(determinar_estacion)
    # Aseguramos que categoria sea de tipo category de pandas
    estacion_df['estacion'] = estacion_df['estacion'].astype('category')
    # Creamos columna estación_semana con la estación predominante en la semana anterior (nd = 7), la estacion es un categoría con string values
    estacion_df['estacion_semana'] = estacion_df['estacion'].rolling(window=7, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    # Creamos columna estación_quincena con la estación predominante en los últimos 15 días (nd = 15)
    estacion_df['estacion_quincena'] = estacion_df['estacion'].rolling(window=15, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    # Creamos columna estación_mes con la estación predominante en el último mes (nd = 30)
    estacion_df['estacion_mes'] = estacion_df['estacion'].rolling(window=30, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    # Pasamos las columnas de estación de [0, 1, 2, 3] a tipo category con ['Verano', 'Otoño', 'Invierno', 'Primavera']
    estacion_df['estacion'] = estacion_df['estacion'].map({0: 'Verano', 1: 'Otoño', 2: 'Invierno', 3: 'Primavera'}).astype('category')
    estacion_df['estacion_semana'] = estacion_df['estacion_semana'].map({0: 'Verano', 1: 'Otoño', 2: 'Invierno', 3: 'Primavera'}).astype('category')
    estacion_df['estacion_quincena'] = estacion_df['estacion_quincena'].map({0: 'Verano', 1: 'Otoño', 2: 'Invierno', 3: 'Primavera'}).astype('category')
    estacion_df['estacion_mes'] = estacion_df['estacion_mes'].map({0: 'Verano', 1: 'Otoño', 2: 'Invierno', 3: 'Primavera'}).astype('category')

    ## Fase lunar
    # Tomamos el dataframe con mayor cantidad de filas (lluvia) como base y nos quedamos con la columna fecha
    lunar_df = lluvia_df_subset[['fecha']].copy()
    # Creamos funcion para determinar la fase lunar según la fecha
    def determinar_fase_lunar(fecha):
        dia = moon.phase(fecha)
        if 0 <= dia <= 6:
            return 0
            # return "Luna Nueva"
        elif 7 <= dia <= 13:
            return 1
            # return "Cuarto Creciente"
        elif 14 <= dia <= 20:
            return 2
            # return "Luna Llena"
        else:  # 21–29
            return 3
            # return "Cuarto Menguante"
    # Creamos la columna fase lunar aplicando la función
    lunar_df['fase_lunar'] = lunar_df['fecha'].apply(determinar_fase_lunar)
    # Aseguramos que categoria sea de tipo category de pandas
    lunar_df['fase_lunar'] = lunar_df['fase_lunar'].astype('category')
    # No vamos a usar este dato par el modelo principal, sino que para el extra, así que no calculamos otros nd. Además, la fase lunar cambia cada 7-8 días, por lo que no tendría sentido.
    # Pasamos las columnas de fase lunar de [0, 1, 2, 3] a tipo category con ['Luna Nueva', 'Cuarto Creciente', 'Luna Llena', 'Cuarto Menguante']
    lunar_df['fase_lunar'] = lunar_df['fase_lunar'].map({0: 'Luna Nueva', 1: 'Cuarto Creciente', 2: 'Luna Llena', 3: 'Cuarto Menguante'}).astype('category')

    ## Merge de todos los dataframes
    dataframes = [lluvia_df_subset, temperatura_df_subset, humedad_df_subset, viento_df_subset, fenomenos_df_subset, estacion_df, lunar_df]
    merged_preliminar_df = reduce(lambda left, right: pd.merge(left, right, on='fecha', how='outer'), dataframes)
    # Ordenamos por fecha
    merged_preliminar_df = merged_preliminar_df.sort_values(by='fecha').reset_index(drop=True)

    ## Separamos el dataset en conjunto de entrenamiento y prueba
    print('Separando conjunto de entrenamiento y prueba...')
    merged_preliminar_X_df = merged_preliminar_df.drop(columns=['manana_llueve'])
    merged_preliminar_y_df = merged_preliminar_df['manana_llueve']
    # Usamos 80% para entrenamiento y 20% para prueba, usando scikit-learn
    entrenamiento_X_df, prueba_X_df, entrenamiento_y_df, prueba_y_df = train_test_split(
        merged_preliminar_X_df, merged_preliminar_y_df,
        test_size=0.2,
        random_state=13,
        shuffle=True,
        stratify=merged_preliminar_y_df
    )
    # Reunimos nuevamente los conjuntos de entrenamiento y prueba
    entrenamiento_df = pd.concat([entrenamiento_X_df, entrenamiento_y_df], axis=1).reset_index(drop=True)
    prueba_df = pd.concat([prueba_X_df, prueba_y_df], axis=1).reset_index(drop=True)
    print(f"    Tamaño del conjunto de entrenamiento: {entrenamiento_df.shape}")
    print(f"    Tamaño del conjunto de prueba: {prueba_df.shape}")

    ## Ahora sí, podemos categorizar las variables que lo requieren, usando los datos del conjunto de entrenamiento para definir los umbrales
    # Conjunto de entrenamiento
    # Lluvia
    # Calculamos el promedio de días lluviosos en el conjunto de entrenamiento
    promedio_dias_lluviosos = entrenamiento_df['dia_lluvioso'].mean()
    # Para semana_lluviosa, quincena_lluviosa y mes_lluviosa, marcamos 1 si la cantidad de días lluviosos es mayor o igual al promedio, y 0 si es menor
    entrenamiento_df['semana_lluviosa'] = (entrenamiento_df['semana_lluviosa'] >= promedio_dias_lluviosos * 7).astype(int)
    entrenamiento_df['quincena_lluviosa'] = (entrenamiento_df['quincena_lluviosa'] >= promedio_dias_lluviosos * 15).astype(int)
    entrenamiento_df['mes_lluvioso'] = (entrenamiento_df['mes_lluvioso'] >= promedio_dias_lluviosos * 30).astype(int)
    # Temperatura
    # Reemplazamos los valores NaN de temperatura por la media de cada columna del conjunto de entrenamiento
    temp_max_promedio = entrenamiento_df['temp_max'].mean()
    temp_min_promedio = entrenamiento_df['temp_min'].mean()
    entrenamiento_df.fillna({'temp_max': temp_max_promedio, 'temp_max_semana': temp_max_promedio, 'temp_max_quincena': temp_max_promedio, 'temp_max_mes': temp_max_promedio}, inplace=True)
    entrenamiento_df.fillna({'temp_min': temp_min_promedio, 'temp_min_semana': temp_min_promedio, 'temp_min_quincena': temp_min_promedio, 'temp_min_mes': temp_min_promedio}, inplace=True)
    # Calculamos la temperatura promedio diaria
    entrenamiento_df['temp_promedio_dia'] = (entrenamiento_df['temp_max'] + entrenamiento_df['temp_min']) / 2
    entrenamiento_df['temp_promedio_semana'] = (entrenamiento_df['temp_max_semana'] + entrenamiento_df['temp_min_semana']) / 2
    entrenamiento_df['temp_promedio_quincena'] = (entrenamiento_df['temp_max_quincena'] + entrenamiento_df['temp_min_quincena']) / 2
    entrenamiento_df['temp_promedio_mes'] = (entrenamiento_df['temp_max_mes'] + entrenamiento_df['temp_min_mes']) / 2 
    # Calculamos los umbrales para categorizar la temperatura
    max_temp = entrenamiento_df['temp_promedio_dia'].max()
    min_temp = entrenamiento_df['temp_promedio_dia'].min()
    umbral_alta = min_temp + (max_temp - min_temp) * 2 / 3
    umbral_baja = min_temp + (max_temp - min_temp) * 1 / 3
    # Creamos funcion para categorizar la temperatura según umbrales en 'Baja', 'Media' y 'Alta'
    def categorizar_temperatura(temp):
        if temp < umbral_baja:
            return 'Baja'
        elif umbral_baja <= temp < umbral_alta:
            return 'Media'
        else:
            return 'Alta'
    # La aplicamos al dataframe y aseguramos que sea de tipo category de pandas
    entrenamiento_df['temp_categoria_dia'] = entrenamiento_df['temp_promedio_dia'].apply(categorizar_temperatura).astype('category')
    entrenamiento_df['temp_categoria_semana'] = entrenamiento_df['temp_promedio_semana'].apply(categorizar_temperatura).astype('category')
    entrenamiento_df['temp_categoria_quincena'] = entrenamiento_df['temp_promedio_quincena'].apply(categorizar_temperatura).astype('category')
    entrenamiento_df['temp_categoria_mes'] = entrenamiento_df['temp_promedio_mes'].apply(categorizar_temperatura).astype('category')
    # Humedad relativa
    # Reemplazamos los valores NaN de humedad relativa por la media de la columna del conjunto de entrenamiento
    humedad_relativa_promedio = entrenamiento_df['humedad_relativa'].mean()
    entrenamiento_df.fillna({'humedad_relativa': humedad_relativa_promedio, 'humedad_relativa_semana': humedad_relativa_promedio, 'humedad_relativa_quincena': humedad_relativa_promedio, 'humedad_relativa_mes': humedad_relativa_promedio}, inplace=True)
    # Calculamos los umbrales para categorizar la humedad relativa en 'Baja', 'Media' y 'Alta'
    max_humedad = entrenamiento_df['humedad_relativa'].max()
    min_humedad = entrenamiento_df['humedad_relativa'].min()
    umbral_alta_hum = min_humedad + (max_humedad - min_humedad) * 2 / 3
    umbral_baja_hum = min_humedad + (max_humedad - min_humedad) * 1 / 3
    # Creamos funcion para categorizar la humedad relativa en 'Baja', 'Media' y 'Alta'
    def categorizar_humedad(hum):
        if hum < umbral_baja_hum:
            return 'Baja'
        elif umbral_baja_hum <= hum < umbral_alta_hum:
            return 'Media'
        else:
            return 'Alta'
    # La aplicamos al dataframe y aseguramos que sea de tipo category de pandas
    entrenamiento_df['humedad_categoria_dia'] = entrenamiento_df['humedad_relativa'].apply(categorizar_humedad).astype('category')
    entrenamiento_df['humedad_categoria_semana'] = entrenamiento_df['humedad_relativa_semana'].apply(categorizar_humedad).astype('category')
    entrenamiento_df['humedad_categoria_quincena'] = entrenamiento_df['humedad_relativa_quincena'].apply(categorizar_humedad).astype('category')
    entrenamiento_df['humedad_categoria_mes'] = entrenamiento_df['humedad_relativa_mes'].apply(categorizar_humedad).astype('category')
    # Viento
    # Reemplazamos los valores NaN de viento por la moda del mes correspondiente (usando el conjunto de entrenamiento)
    # Para ello, extraemos el mes de la fecha y usamos groupby para obtener la moda por mes
    entrenamiento_df['mes'] = entrenamiento_df['fecha'].dt.month
    moda_viento_por_mes = entrenamiento_df.groupby('mes')['direccion_viento'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    # Consideramos también mes de la semana, quincena y mes como la moda del mes
    entrenamiento_df['mes_semana'] = entrenamiento_df['mes'].rolling(window=7, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    entrenamiento_df['mes_quincena'] = entrenamiento_df['mes'].rolling(window=15, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    entrenamiento_df['mes_mes'] = entrenamiento_df['mes'].rolling(window=30, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    # Función para reemplazar NaN por la moda del mes
    def reemplazar_viento_por_moda(row):
        # Si no es NaN en ninguna de las columnas de dirección del viento, no hacemos nada
        if not pd.isna(row['direccion_viento']) and not pd.isna(row['direccion_viento_semana']) and not pd.isna(row['direccion_viento_quincena']) and not pd.isna(row['direccion_viento_mes']):
            return row
        # Si es NaN en direccion_viento, lo reemplazamos por la moda del mes
        else:
            if pd.isna(row['direccion_viento']):
                row['direccion_viento'] = moda_viento_por_mes[row['mes']]
            if pd.isna(row['direccion_viento_semana']):
                row['direccion_viento_semana'] = moda_viento_por_mes[row['mes_semana']]
            if pd.isna(row['direccion_viento_quincena']):
                row['direccion_viento_quincena'] = moda_viento_por_mes[row['mes_quincena']]
            if pd.isna(row['direccion_viento_mes']):
                row['direccion_viento_mes'] = moda_viento_por_mes[row['mes_mes']]
            return row
    # Aplicamos la función al dataframe
    entrenamiento_df = entrenamiento_df.apply(reemplazar_viento_por_moda, axis=1)
    # Eliminamos las columnas auxiliares de mes
    entrenamiento_df.drop(columns=['mes', 'mes_semana', 'mes_quincena', 'mes_mes'], inplace=True)
    # Aseguramos que categoria sea de tipo category de pandas
    entrenamiento_df['direccion_viento'] = entrenamiento_df['direccion_viento'].astype('category')
    # Truenos
    # Reemplazamos los valores NaN de hay_truenos por 0 (no hubo truenos)
    entrenamiento_df['hay_truenos'] = entrenamiento_df['hay_truenos'].fillna(0).astype(int)
    entrenamiento_df['hay_truenos_semana'] = entrenamiento_df['hay_truenos_semana'].fillna(0).astype(int)
    entrenamiento_df['hay_truenos_quincena'] = entrenamiento_df['hay_truenos_quincena'].fillna(0).astype(int)
    entrenamiento_df['hay_truenos_mes'] = entrenamiento_df['hay_truenos_mes'].fillna(0).astype(int)
    # Calculamos la media de truenos en el conjunto de entrenamiento
    promedio_truenos = entrenamiento_df['hay_truenos'].mean()
    # Si hubo truenos más días que el promedio, marcamos 1, sino 0
    entrenamiento_df['hay_truenos'] = (entrenamiento_df['hay_truenos'] >= promedio_truenos).astype(int)
    entrenamiento_df['hay_truenos_semana'] = (entrenamiento_df['hay_truenos_semana'] >= promedio_truenos * 7).astype(int)
    entrenamiento_df['hay_truenos_quincena'] = (entrenamiento_df['hay_truenos_quincena'] >= promedio_truenos * 15).astype(int)
    entrenamiento_df['hay_truenos_mes'] = (entrenamiento_df['hay_truenos_mes'] >= promedio_truenos * 30).astype(int)

    # Conjunto de prueba
    # Lluvia
    prueba_df['semana_lluviosa'] = (prueba_df['semana_lluviosa'] >= promedio_dias_lluviosos * 7).astype(int)
    prueba_df['quincena_lluviosa'] = (prueba_df['quincena_lluviosa'] >= promedio_dias_lluviosos * 15).astype(int)
    prueba_df['mes_lluvioso'] = (prueba_df['mes_lluvioso'] >= promedio_dias_lluviosos * 30).astype(int)
    # Temperatura
    prueba_df.fillna({'temp_max': temp_max_promedio, 'temp_max_semana': temp_max_promedio, 'temp_max_quincena': temp_max_promedio, 'temp_max_mes': temp_max_promedio}, inplace=True)
    prueba_df.fillna({'temp_min': temp_min_promedio, 'temp_min_semana': temp_min_promedio, 'temp_min_quincena': temp_min_promedio, 'temp_min_mes': temp_min_promedio}, inplace=True)
    prueba_df['temp_promedio_dia'] = (prueba_df['temp_max'] + prueba_df['temp_min']) / 2
    prueba_df['temp_promedio_semana'] = (prueba_df['temp_max_semana'] + prueba_df['temp_min_semana']) / 2
    prueba_df['temp_promedio_quincena'] = (prueba_df['temp_max_quincena'] + prueba_df['temp_min_quincena']) / 2
    prueba_df['temp_promedio_mes'] = (prueba_df['temp_max_mes'] + prueba_df['temp_min_mes']) / 2
    prueba_df['temp_categoria_dia'] = prueba_df['temp_promedio_dia'].apply(categorizar_temperatura).astype('category')
    prueba_df['temp_categoria_semana'] = prueba_df['temp_promedio_semana'].apply(categorizar_temperatura).astype('category')
    prueba_df['temp_categoria_quincena'] = prueba_df['temp_promedio_quincena'].apply(categorizar_temperatura).astype('category')
    prueba_df['temp_categoria_mes'] = prueba_df['temp_promedio_mes'].apply(categorizar_temperatura).astype('category')
    # Humedad relativa
    prueba_df.fillna({'humedad_relativa': humedad_relativa_promedio, 'humedad_relativa_semana': humedad_relativa_promedio, 'humedad_relativa_quincena': humedad_relativa_promedio, 'humedad_relativa_mes': humedad_relativa_promedio}, inplace=True)
    prueba_df['humedad_categoria_dia'] = prueba_df['humedad_relativa'].apply(categorizar_humedad).astype('category')
    prueba_df['humedad_categoria_semana'] = prueba_df['humedad_relativa_semana'].apply(categorizar_humedad).astype('category')
    prueba_df['humedad_categoria_quincena'] = prueba_df['humedad_relativa_quincena'].apply(categorizar_humedad).astype('category')
    prueba_df['humedad_categoria_mes'] = prueba_df['humedad_relativa_mes'].apply(categorizar_humedad).astype('category')
    # Viento
    prueba_df['mes'] = prueba_df['fecha'].dt.month
    prueba_df['mes_semana'] = prueba_df['mes'].rolling(window=7, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    prueba_df['mes_quincena'] = prueba_df['mes'].rolling(window=15, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    prueba_df['mes_mes'] = prueba_df['mes'].rolling(window=30, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    prueba_df = prueba_df.apply(reemplazar_viento_por_moda, axis=1)
    prueba_df.drop(columns=['mes', 'mes_semana', 'mes_quincena', 'mes_mes'], inplace=True)
    prueba_df['direccion_viento'] = prueba_df['direccion_viento'].astype('category')
    # Truenos
    prueba_df['hay_truenos'] = prueba_df['hay_truenos'].fillna(0).astype(int)
    prueba_df['hay_truenos_semana'] = prueba_df['hay_truenos_semana'].fillna(0).astype(int)
    prueba_df['hay_truenos_quincena'] = prueba_df['hay_truenos_quincena'].fillna(0).astype(int)
    prueba_df['hay_truenos_mes'] = prueba_df['hay_truenos_mes'].fillna(0).astype(int)

    # Nos quedamos solo con las columnas que nos interesan
    columnas_interes_m1 = ['fecha', 'dia_lluvioso', 'temp_categoria_dia', 'humedad_categoria_dia', 'direccion_viento', 'estacion', 'manana_llueve']
    columnas_interes_m2 = columnas_interes_m1 + ['hay_truenos', 'fase_lunar']
    entrenamiento_final_m1_df = entrenamiento_df[columnas_interes_m1].copy()
    prueba_final_m1_df = prueba_df[columnas_interes_m1].copy()
    entrenamiento_final_m2_df = entrenamiento_df[columnas_interes_m2].copy()
    prueba_final_m2_df = prueba_df[columnas_interes_m2].copy()
    # Para modelos de Naive Bayes, creamos tambien versiones con distintos valores de nd (7, 15, 30) para las variables categóricas
    # Modelo 1
    columnas_interes_m1_7 = ['fecha', 'semana_lluviosa', 'temp_categoria_semana', 'humedad_categoria_semana', 'direccion_viento_semana', 'estacion_semana', 'manana_llueve']
    columnas_interes_m1_15 = ['fecha', 'quincena_lluviosa', 'temp_categoria_quincena', 'humedad_categoria_quincena', 'direccion_viento_quincena', 'estacion_quincena', 'manana_llueve']
    columnas_interes_m1_30 = ['fecha', 'mes_lluvioso', 'temp_categoria_mes', 'humedad_categoria_mes', 'direccion_viento_mes', 'estacion_mes', 'manana_llueve']
    entrenamiento_final_m1_7_df = entrenamiento_df[columnas_interes_m1_7].copy()
    prueba_final_m1_7_df = prueba_df[columnas_interes_m1_7].copy()
    entrenamiento_final_m1_15_df = entrenamiento_df[columnas_interes_m1_15].copy()
    prueba_final_m1_15_df = prueba_df[columnas_interes_m1_15].copy()
    entrenamiento_final_m1_30_df = entrenamiento_df[columnas_interes_m1_30].copy()
    prueba_final_m1_30_df = prueba_df[columnas_interes_m1_30].copy()

    print("Preprocesamiento finalizado.")
    print("Datasets generados:")
    print(f" - Dataset: Entrenamiento {entrenamiento_final_m1_df.shape}, Prueba {prueba_final_m1_df.shape} - columnas: {entrenamiento_final_m1_df.columns.tolist()}")
    print(f" - Dataset con nd=7: Entrenamiento {entrenamiento_final_m1_7_df.shape}, Prueba {prueba_final_m1_7_df.shape} - columnas: {entrenamiento_final_m1_7_df.columns.tolist()}")
    print(f" - Dataset con nd=15: Entrenamiento {entrenamiento_final_m1_15_df.shape}, Prueba {prueba_final_m1_15_df.shape} - columnas: {entrenamiento_final_m1_15_df.columns.tolist()}")
    print(f" - Dataset con nd=30: Entrenamiento {entrenamiento_final_m1_30_df.shape}, Prueba {prueba_final_m1_30_df.shape} - columnas: {entrenamiento_final_m1_30_df.columns.tolist()}")
    print(f" - Dataset bonus: Entrenamiento {entrenamiento_final_m2_df.shape}, Prueba {prueba_final_m2_df.shape} - columnas: {entrenamiento_final_m2_df.columns.tolist()}")


    ###########################
    ### Arboles de decision ###
    ###########################
    print("\nModelado con árbol de decisión")

    ## Cross Validation (set entrenamiento)
    print("Realizando cross validation para optimizar hiperparámetros...")
    # En esta celda se realiza cross validation sobre el conjunto de entrenamiento para encontrar el mejor hiperparametro de min_info_gain, y tambien la profundida del arbol asociada.
    # Seleccionamos el modelo 2 (con más variables)
    X_train = entrenamiento_final_m1_df.drop(columns=['fecha', 'manana_llueve'])
    y_train = entrenamiento_final_m1_df['manana_llueve']
    X_test = prueba_final_m1_df.drop(columns=['fecha', 'manana_llueve'])
    y_test = prueba_final_m1_df['manana_llueve']
    # Convertimos variables categóricas a números (label encoding)
    for col in X_train.columns:
        if X_train[col].dtype.name == 'category' or X_train[col].dtype == object:
            cats = X_train[col].astype('category').cat.categories
            X_train[col] = X_train[col].astype('category').cat.codes
            X_test[col] = X_test[col].astype('category').cat.set_categories(cats).cat.codes
    # Función para entrenar y evaluar el árbol de decisión
    max_profundidades = [1, 3, 5, 7, 9, 11, 13]
    min_info_gains = [0.001, 0.005, 0.01, 0.03, 0.1]
    mejor_fbeta = 0
    mejor_params = None
    resultados = []
    for mig in min_info_gains:
        for md in max_profundidades:
            kf = KFold(n_splits=6, shuffle=True, random_state=26)
            recalls = []
            precisions = []
            accuracies = []
            fbeta_scores = []
            for train_idx, test_idx in kf.split(X_train):
                X_tr, X_te = X_train.iloc[train_idx].values, X_train.iloc[test_idx].values
                y_tr, y_te = y_train.iloc[train_idx].values, y_train.iloc[test_idx].values
                acc, prec, rec, fbeta, cm, y_pred = entrenar_y_evaluar_arbol(X_tr, y_tr, X_te, y_te, mig, md)
                recalls.append(rec)
                precisions.append(prec)
                accuracies.append(acc)
                fbeta_scores.append(fbeta)
            mean_recall = np.mean(recalls)
            mean_precision = np.mean(precisions)
            mean_accuracy = np.mean(accuracies)
            mean_fbeta = np.mean(fbeta_scores)
            std_recall = np.std(recalls)
            resultados.append((md, mig, mean_recall, std_recall, mean_precision, mean_accuracy, mean_fbeta))
            if mean_fbeta > mejor_fbeta:
                mejor_fbeta = mean_fbeta
                mejor_params = (md, mig)
    #
    # cv_results_df = pd.DataFrame(resultados, columns=["max_depth", "main_info_gain", "mean_recall", "std_recall", "mean_precision", "mean_accuracy", "mean_fbeta"])
    print(f"    Mejor combinación: Profundidad={mejor_params[0]}, main_info_gain={mejor_params[1]}, se obtiene Fbeta Score={mejor_fbeta:.3f}")

    # En esta celda se realiza el algoritmo de chi-cuadrado, para determinar una seleccion de atributos de nuestro dataset. Se recorre un rango de 2 a 6 atributos, con los datos obtenidos, utilizadno los valores encontrados de hiperparametros de cross validation.
    print("Realizando selección de atributos con Chi-cuadrado...")

    # Usamos 80% para entrenamiento y 20% para prueba, usando scikit-learn
    entrenamiento_chi_cuadrado_x, prueba_chi_cuadrado_x, entrenamiento_y_chi_cuadrado, prueba_y_chi_cuadrado = sk.model_selection.train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=13,
    shuffle=True,
    stratify=y_train
    )
    min_info_gain = 0.01
    max_profundidad = 7
    atr_mejor = None
    atr_mejor_recall = 0
    atr_mejor_fbeta = 0
    atr_mejor_prec = 0
    atr_mejor_acc = 0
    atr_mejor_cm = None
    for k in range(2, 6):
        selector = SelectKBest(score_func=chi2, k=k)
        X_train_selected = selector.fit_transform(entrenamiento_chi_cuadrado_x, entrenamiento_y_chi_cuadrado)
        X_test_selected = selector.transform(prueba_chi_cuadrado_x)
        # Ver qué columnas fueron seleccionadas:
        selected_columns = X_train.columns[selector.get_support()]
        acc, prec, rec, fbeta, cm, y_pred = entrenar_y_evaluar_arbol(X_train_selected, entrenamiento_y_chi_cuadrado.values, X_test_selected, prueba_y_chi_cuadrado.values, min_info_gain, max_profundidad)
        if fbeta > atr_mejor_fbeta:
            atr_mejor_fbeta = fbeta
            atr_mejor_recall = rec
            atr_mejor_prec = prec
            atr_mejor_acc = acc
            atr_mejor_cm = cm
            atr_mejor = list(selected_columns)
    print(f"    Mejor combinación de atributos: {atr_mejor}, se obtiene Fbeta Score={atr_mejor_fbeta:.3f}")


    print("Entrenando y evaluando con mejor combinación de atributos e hiperparametros...")
    X2_train = entrenamiento_final_m1_df.drop(columns=['fecha', 'manana_llueve'])
    y2_train = entrenamiento_final_m1_df['manana_llueve']
    X2_test = prueba_final_m1_df.drop(columns=['fecha', 'manana_llueve'])
    y2_test = prueba_final_m1_df['manana_llueve']
    min_info_gain = 0.01
    max_profundidad = 7
    # Convertimos variables categóricas a números (label encoding)
    for col in X2_train.columns:
        if X2_train[col].dtype.name == 'category' or X2_train[col].dtype == object:
            cats = X2_train[col].astype('category').cat.categories
            X2_train[col] = X2_train[col].astype('category').cat.codes
            X2_test[col] = X2_test[col].astype('category').cat.set_categories(cats).cat.codes
    # Entrenamiento y evaluación
    acc, prec, rec, f1, cm, y_pred = entrenar_y_evaluar_arbol(
        X2_train.values, y2_train.values, X2_test.values, y2_test.values, min_info_gain, max_profundidad
    )
    print(f"    Accuracy: {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall: {rec:.4f}")
    print(f"    F1 Score: {f1:.4f}")
    print("    Matriz de confusión:")
    print("    ", cm[0])
    print("    ", cm[1])

    # ## Random Forest
    print("Comparando con modelado con Random Forest (vs mejor combinación de atributos)...")
    # En esta etapa, se realiza entrenamiento y prueba, utilizando el algoritmo de Random Forest Classifier proporcionado por sklearn.
    X_train_rf = X_train.copy()
    X_test_rf = X_test.copy()
    rf = RandomForestClassifier(random_state=26, class_weight='balanced', criterion='entropy', min_impurity_decrease=0.01, n_estimators=100, max_depth=7)
    rf.fit(X_train_rf, y_train)
    # Predecimos
    y_pred_rf = rf.predict(X_test_rf)
    # Métricas
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, average='macro')
    rec_rf = recall_score(y_test, y_pred_rf, average='macro')
    fbeta_rf = fbeta_score(y_test, y_pred_rf,beta = 2, average='macro')
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print(f"    RandomForest - Accuracy: {acc_rf:.4f} ({(acc_rf/atr_mejor_acc * 100) - 100:+.1f}%)")
    print(f"    RandomForest - Precision: {prec_rf:.4f} ({(prec_rf/atr_mejor_prec * 100) - 100:+.1f}%)")
    print(f"    RandomForest - Recall: {rec_rf:.4f} ({(rec_rf/atr_mejor_recall * 100) - 100:+.1f}%)")
    print(f"    RandomForest - Fbeta Score: {fbeta_rf:.4f} ({(fbeta_rf/atr_mejor_fbeta * 100) - 100:+.1f}%)")
    print("    RandomForest - Matriz de confusión:")
    print("    ", atr_mejor_cm[0])
    print("    ", atr_mejor_cm[1])


    ###################
    ### Naive Bayes ###
    ###################
    print("\nModelado con Naive Bayes")

    # ### Selección de Hiperparametros
    # Inicializar variables para almacenar los mejores resultados
    k_max=-1
    m_max=-1
    n_max=-1
    f2_max=-1
    #Lista utilizada para graficar
    datos_f2=[]
    datos_rec=[]
    datos_pres=[]
    # Iterar segun valores de nd
    print("Realizando selección de hiperparámetros (k, m, n)...")
    for entrenamiento_NB_df, prueba_NB_df, n in [(entrenamiento_final_m1_df, prueba_final_m1_df, 1), (entrenamiento_final_m1_7_df, prueba_final_m1_7_df, 7), (entrenamiento_final_m1_15_df, prueba_final_m1_15_df, 15), (entrenamiento_final_m1_30_df, prueba_final_m1_30_df, 30)]:
        valores_m=[1, 3, 5, 10, 15]
        valores_k=[2, 3, 4, 5]
        y_entrenamiento_NB = entrenamiento_NB_df['manana_llueve']
        entrenamiento_NB_df = entrenamiento_NB_df.drop(columns=['fecha', "manana_llueve"]).copy()
        y_prueba_NB = prueba_NB_df['manana_llueve']
        prueba_NB_df = prueba_NB_df.drop(columns=['fecha', "manana_llueve"]).copy()
        # Convertir todos los atributos categóricos a tipo 'category'
        for col in entrenamiento_NB_df.columns:
                entrenamiento_NB_df[col] = entrenamiento_NB_df[col].astype('category')
        entrenamiento_NB_df = entrenamiento_NB_df.map(encode)
        prueba_NB_df = prueba_NB_df.map(encode)
        # Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=86)
        for m in valores_m:
            for k in valores_k:
                valores_f2= []
                valores_pres = []
                valores_rec = []
                valores_acc = []
                # Particionamos los datos
                for train_index, val_index in kf.split(entrenamiento_NB_df):    
                    X_train_fold = entrenamiento_NB_df.iloc[train_index]
                    X_val_fold = entrenamiento_NB_df.iloc[val_index]
                    y_train_fold = y_entrenamiento_NB.iloc[train_index]
                    y_val_fold = y_entrenamiento_NB.iloc[val_index]
                    selector = SelectKBest(chi2, k=k)
                    selector.fit(X_train_fold, y_train_fold)
                    # Aplicar selección a train y validation de este fold
                    X_train_selec = X_train_fold.iloc[:, selector.get_support()]
                    X_val_selec = X_val_fold.iloc[:, selector.get_support()]
                    # Aplicamos Random Oversampling
                    oversampler = RandomOverSampler(random_state=3)
                    X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train_selec, y_train_fold)
                    # Entrenamos el modelo
                    modelo_nb_cv = NaiveBayes(m)
                    modelo_nb_cv.fit(X_train_balanced, y_train_balanced)
                    # Evaluamos
                    predicciones_cv = modelo_nb_cv.predict(X_val_selec)
                    # Calculamos las métricas
                    valores_acc.append(accuracy_score(y_val_fold, predicciones_cv))
                    pres=precision_score(y_val_fold, predicciones_cv)
                    valores_pres.append(pres)
                    rec=recall_score(y_val_fold, predicciones_cv)
                    valores_rec.append(rec)
                    f2=fbeta_score(y_val_fold, predicciones_cv, beta=2)
                    valores_f2.append(f2)
                    if(n==7):
                        datos_pres.append((m,k,pres))
                        datos_rec.append((m,k,rec))
                        datos_f2.append((m, k, f2))
                precision_promedio = np.mean(valores_pres)
                recall_promedio = np.mean(valores_rec)
                f2_promedio = np.mean(valores_f2)
                accuracy_promedio = np.mean(valores_acc)
                k_max, m_max, n_max, f2_max = (k, m, n, f2_promedio) if f2_promedio > f2_max else (k_max, m_max, n_max, f2_max)
    print(f"    Mejor modelo Naive Bayes: Variables con nd={n_max}, k={k_max}, m={m_max}, se obtiene F2 Score={f2_max:.3f}")

    # ### Entrenamiento completo
    # #### Se entrena al modelo con los hiperparametros optimos obtenidos en el paso anterior y todo el conjunto de entrenamiento, se valida contra el conjunto de pruebas apartado al inicio.
    print("Entrenando modelo final con mejores hiperparámetros...")
    # Entrenamos en base a los hiperparametros óptimos encontrados
    m=1
    k=2
    # nd=7
    entrenamiento_NB_df = entrenamiento_final_m1_7_df.copy()
    prueba_NB_df = prueba_final_m1_7_df.copy()
    # Preprocesamiento
    y_entrenamiento_NB = entrenamiento_NB_df['manana_llueve']
    entrenamiento_NB_df = entrenamiento_NB_df.drop(columns=['fecha', "manana_llueve"]).copy().map(encode)
    y_prueba_NB = prueba_NB_df['manana_llueve']
    prueba_NB_df = prueba_NB_df.drop(columns=['fecha', "manana_llueve"]).copy().map(encode) 
    # Seleccion de atributos
    selector = SelectKBest(chi2, k=k)
    selector.fit(entrenamiento_NB_df, y_entrenamiento_NB)
    entrenamiento_NB_df = entrenamiento_NB_df.iloc[:, selector.get_support()]
    prueba_NB_df = prueba_NB_df.iloc[:, selector.get_support()]
    # Aplicamos Random Oversampling (50 - 50)
    oversampler = RandomOverSampler(random_state=47)
    entrenamiento_NB_df, y_entrenamiento_NB = oversampler.fit_resample(entrenamiento_NB_df, y_entrenamiento_NB)
    #print("\nDistribución después de Random Oversampling :", y_entrenamiento_NB.value_counts())
    # Entrenamos
    modelo_nb = NaiveBayes(m=m)
    modelo_nb.fit(entrenamiento_NB_df, y_entrenamiento_NB)
    # Evaluamos
    predicciones_cv = modelo_nb.predict(prueba_NB_df)
    accuracy_nb = accuracy_score(y_prueba_NB, predicciones_cv)
    precision_nb = precision_score(y_prueba_NB, predicciones_cv)
    recall_nb = recall_score(y_prueba_NB, predicciones_cv)
    f2_nb = fbeta_score(y_prueba_NB, predicciones_cv, beta=2)
    print(f"    Accuracy: {accuracy_nb:.4f}")
    print(f"    Precisión: {precision_nb:.4f}")
    print(f"    Recall: {recall_nb:.4f}")
    print(f"    F2-Score: {f2_nb:.4f}")
    # Matriz de confusion
    mc = confusion_matrix(y_prueba_NB, predicciones_cv)
    print("    Matriz de confusión:")
    print("    ", mc[0])
    print("    ", mc[1])

    # ### CategoricalNB
    print("Comparando con CategoricalNB de sklearn (vs modelo con mejores hiperparámetros)...")
    # Naive Bayes con CategoricalNB
    entrenamiento_encoded_NB_df = entrenamiento_final_m1_df.drop(columns=['fecha', 'manana_llueve']).copy().map(encode)
    y_entrenamiento_NB_df = entrenamiento_final_m1_df['manana_llueve'].copy()
    prueba_encoded_NB_df = prueba_final_m1_df.drop(columns=['fecha', 'manana_llueve']).copy().map(encode)
    y_prueba_NB_df = prueba_final_m1_df['manana_llueve'].copy()

    # Seleccion de atributos
    selector = SelectKBest(chi2, k=3)
    selector.fit(entrenamiento_encoded_NB_df, y_entrenamiento_NB_df)
    entrenamiento_encoded_NB_df = entrenamiento_encoded_NB_df.iloc[:, selector.get_support()]
    prueba_encoded_NB_df = prueba_encoded_NB_df.iloc[:, selector.get_support()]

    # Creamos el modelo
    nb_model = CategoricalNB(alpha=100, fit_prior=False)
    # Entrenamos el modelo
    nb_model.fit(entrenamiento_encoded_NB_df, y_entrenamiento_NB_df)
    # Hacemos predicciones
    predicciones = nb_model.predict(prueba_encoded_NB_df)
    # Evaluamos el modelo
    precision_cnb = precision_score(y_prueba_NB_df, predicciones)
    recall_cnb = recall_score(y_prueba_NB_df, predicciones)
    f2_cnb = fbeta_score(y_prueba_NB_df, predicciones, beta=2)
    accuracy_cnb = accuracy_score(y_prueba_NB_df, predicciones)

    print(f"    Resultados CategoricalNB:")
    print(f"    Accuracy: {accuracy_cnb:.4f} ({(accuracy_cnb/accuracy_nb * 100) - 100:+.1f}%)")
    print(f"    Precisión: {precision_cnb:.4f} ({(precision_cnb/precision_nb * 100) - 100:+.1f}%)")
    print(f"    Recall: {recall_cnb:.4f} ({(recall_cnb/recall_nb * 100) - 100:+.1f}%)")
    print(f"    F2-Score: {f2_cnb:.4f} ({(f2_cnb/f2_nb * 100) - 100:+.1f}%)")
    mc_cnb = confusion_matrix(y_prueba_NB_df, predicciones)
    print("    Matriz de confusión CategoricalNB:")
    print("    ", mc_cnb[0])
    print("    ", mc_cnb[1])

    # Pregunta bonus
    print("Realizando análisis pregunta bonus...")
    m=1
    entrenamiento_NB_df = entrenamiento_final_m2_df.copy()
    prueba_NB_df = prueba_final_m2_df.copy()
    # Preprocesamiento
    y_entrenamiento_NB = entrenamiento_NB_df['manana_llueve']
    entrenamiento_NB_df = entrenamiento_NB_df[['hay_truenos', "fase_lunar"]].copy().map(encode)
    y_prueba_NB = prueba_NB_df['manana_llueve']
    prueba_NB_df = prueba_NB_df[['hay_truenos', "fase_lunar"]].copy().map(encode)

    # Aplicamos Random Oversampling (50 - 50)
    oversampler = RandomOverSampler(random_state=27)
    entrenamiento_NB_df, y_entrenamiento_NB = oversampler.fit_resample(entrenamiento_NB_df, y_entrenamiento_NB)
    #print("\nDistribución después de Random Oversampling :", y_entrenamiento_NB.value_counts())

    # Entrenamos
    modelo_nb = NaiveBayes(m=m)
    modelo_nb.fit(entrenamiento_NB_df, y_entrenamiento_NB)

    # Evaluamos
    predicciones_cv = modelo_nb.predict(prueba_NB_df)

    accuracy = accuracy_score(y_prueba_NB, predicciones_cv)
    precision = precision_score(y_prueba_NB, predicciones_cv)
    recall = recall_score(y_prueba_NB, predicciones_cv)
    f2 = fbeta_score(y_prueba_NB, predicciones_cv, beta=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F2-Score: {f2:.4f}")

    # Matriz de confusion
    mc = confusion_matrix(y_prueba_NB, predicciones_cv)
    print("    Matriz de confusión:")
    print("    ", mc[0])
    print("    ", mc[1])

    frase = modelo_nb.predict(pd.DataFrame([{"hay_truenos": 1, "fase_lunar": "Luna Nueva"}]).map(encode))
    print("Truenos con luna nueva, prepárese a que llueva.", "Verdadero" if frase.values[0] else "Falso")


if __name__ == "__main__":
    main()
