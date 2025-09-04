import pandas as pd

archivo_truenos = './Dataset_INUMET/CSV/Fenomenos.csv'

try:
    df_truenos = pd.read_csv(archivo_truenos , sep=';')
    print("DataFrame cargado con éxito desde el archivo CSV")
except FileNotFoundError:
    print(f"Error: El archivo '{archivo_truenos}' no se encontró.")
    exit()
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")
    exit()

# Asumo que la primera columna es la fecha.
# Aca renombro para que me quede mas facil ubicar y no por el nombre largo.

# Primero, veo los datos originales para entender el formato
print("Datos originales de la primera columna:")
print(df_truenos.iloc[:5, 0])
print(f"Tipo de datos: {df_truenos.iloc[:5, 0].dtype}")

if df_truenos.columns[0] is not None:
     # Usar format='mixed' para manejar diferentes formatos de fecha
     df_truenos[df_truenos.columns[0]] = pd.to_datetime(df_truenos[df_truenos.columns[0]], format='mixed', dayfirst=True)
     df_truenos.rename(columns={df_truenos.columns[0]: 'Fecha'}, inplace=True)
else:
    print("Advertencia: La primera columna no tiene nombre y no se pudo renombrar a 'Fecha'.")
    df_truenos.columns.values[0] = 'Fecha' # Asignar nombre directamente
    df_truenos['Fecha'] = pd.to_datetime(df_truenos['Fecha'], format='mixed', dayfirst=True)

# me quedo con la primera columna y la segunda
df_truenos = df_truenos.iloc[:, :2].copy()
print(df_truenos.head())
print("Verificando valores nulos en el DataFrame:")
print(df_truenos.isnull().sum())

# Para los valores de la segunda columna que son str, los convierto a NaN
df_truenos[df_truenos.columns[1]] = pd.to_numeric(df_truenos[df_truenos.columns[1]], errors='coerce')

# Extraer solo la fecha (sin hora) para agrupar por día
df_truenos['FechaSolo'] = df_truenos['Fecha'].dt.date

# print("DataFrame original columna FechaSolo:")
# print(df_truenos.head())

# Creo una columna que indique si hubo tormenta (17) o no en cada registro
df_truenos['Hay_Truenos'] = (df_truenos[df_truenos.columns[1]] == 17).astype(int)

# Agrupar por día y determinar si hubo al menos una tormenta ese día
df_truenos_procesado = df_truenos.groupby('FechaSolo')['Hay_Truenos'].max().reset_index()
df_truenos_procesado.rename(columns={'FechaSolo': 'Fecha'}, inplace=True)

print("DataFrame procesado - Tormentas por día:")
print(df_truenos_procesado.head(10))
print(f"\nTotal de días procesados: {len(df_truenos_procesado)}")
print(f"Días con tormenta: {df_truenos_procesado['Hay_Truenos'].sum()}")
print(f"Días sin tormenta: {len(df_truenos_procesado) - df_truenos_procesado['Hay_Truenos'].sum()}")

# guardar el DataFrame final a un nuevo archivo CSV, para verlo mejor de manera visual.
df_truenos_procesado.to_csv('./Dataset_INUMET/CSV/Datos_procesados/Truenos_Procesado.csv', sep=';', index=False)