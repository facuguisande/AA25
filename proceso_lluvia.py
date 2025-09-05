import pandas as pd

#cargo el achivo csv con los datos
archivo_lluvias = './Dataset_INUMET/CSV/Lluvia.csv'

try:
    df_lluvia = pd.read_csv(archivo_lluvias, sep=';')
    print("DataFrame cargado con éxito desde el archivo CSV")
except FileNotFoundError:
    print(f"Error: El archivo '{archivo_lluvias}' no se encontró.")
    exit()
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")
    exit()

# me quedo solamente con la columna 3 y la 4
df_lluvia_subset = df_lluvia.iloc[:, [2, 3]].copy()
df_lluvia_subset.columns = ['Fecha', 'Lluvia_Obs1']

# Convertir la columna de fecha a datetime
df_lluvia_subset['Fecha'] = pd.to_datetime(df_lluvia_subset['Fecha'], dayfirst=True)

print(df_lluvia_subset.head())

print("Verificando valores nulos en el DataFrame:")
print(df_lluvia_subset.isnull().sum())
# verifico si hay valores no numericos en lluvia y los reemplazo por 0, por lo que comento el profe
print("Verificando valores no numéricos en la columna 'Lluvia_Obs1':")
df_lluvia_subset['Lluvia_Obs1'] = pd.to_numeric(df_lluvia_subset['Lluvia_Obs1'], errors='coerce').fillna(0)


#para los valores nulos de lluvia, me quedo con la media
df_lluvia_subset['Lluvia_Obs1'].fillna(df_lluvia_subset['Lluvia_Obs1'].mean())

# print(df_lluvia_subset.isnull().sum())
umbral_lluvia = 1.0
df_lluvia_subset['Dia_Lluvioso'] = (df_lluvia_subset['Lluvia_Obs1'] >= umbral_lluvia).astype(int)

print(df_lluvia_subset.head())
# distribucion de dias lluviosos, para poder analizar luego, no es necesario
# pero viene bien tenerlo en cuenta, para cuando ajustemos.
print("Distribución de días lluviosos:")
print(df_lluvia_subset['Dia_Lluvioso'].value_counts())

# guarda el DataFrame procesado en un nuevo archivo CSV, esto solamente para verlo de forma visual
df_lluvia_subset.to_csv('./Dataset_INUMET/CSV/Datos_procesados/Lluvia_procesado.csv', sep=';', index=False)

