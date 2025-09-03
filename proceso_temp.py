import pandas as pd
import matplotlib.pyplot as plt

# Cargo el archivo CSV con los datos de temperatura
temp_csv = './Dataset_INUMET/CSV/Temperatura.csv'

try:
    df_temp = pd.read_csv(temp_csv , sep=';')
    print("DataFrame cargado con éxito desde el archivo CSV")
except FileNotFoundError:
    print(f"Error: El archivo '{temp_csv}' no se encontró.")
    exit()
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")
    exit()

# Asumo que la primera columna es la fecha.
# Aca renombro para que me quede mas facil ubicar y no por el nombre largo.
if df_temp.columns[0] is not None:
     df_temp[df_temp.columns[0]] = pd.to_datetime(df_temp[df_temp.columns[0]], dayfirst=True)
     df_temp.rename(columns={df_temp.columns[0]: 'Fecha'}, inplace=True)
else:
    print("Advertencia: La primera columna no tiene nombre y no se pudo renombrar a 'Fecha'.")
    df_temp.columns.values[0] = 'Fecha' # Asignar nombre directamente
    df_temp['Fecha'] = pd.to_datetime(df_temp['Fecha'], dayfirst=True)

print("Verificando valores nulos en el DataFrame:")
print(df_temp.isnull().sum())

# Creo un nuevo dataset con las 3 primeras columnas
df_temp_subset = df_temp.iloc[:, :3].copy()
print(df_temp_subset.head())

df_temp_subset.rename(columns={df_temp_subset.columns[1]: 'Tmax_Obs1',
                   df_temp_subset.columns[2]: 'Tmin_Obs1'}, inplace=True)

# reemplazar valores nulos por la media
df_temp_subset['Tmax_Obs1'].fillna(df_temp_subset['Tmax_Obs1'].mean())
df_temp_subset['Tmin_Obs1'].fillna(df_temp_subset['Tmin_Obs1'].mean())


df_temp_subset['Temperatura_promedio_dia'] = (df_temp_subset['Tmax_Obs1'] + df_temp_subset['Tmin_Obs1']) / 2
print(df_temp_subset.head())

umbral_alta = 25
umbral_baja = 10

# categorizo las temperatura promedio,mediante los umbrales definidos
def categorizar_temperatura(temp):
    if temp < umbral_baja:
        return 'Baja'
    elif umbral_baja <= temp < umbral_alta:
        return 'Templado'
    else:
        return 'Alta'

df_temp_subset['Temp_Categoria_dia'] = df_temp_subset['Temperatura_promedio_dia'].apply(categorizar_temperatura)
print(df_temp_subset.head())

# Graficar para saber la distirbucion de las temperaturas, solo para fines de observar nomas

# plt.figure(figsize=(12, 6))
# plt.plot(df_temp_subset['Fecha'], df_temp_subset['Temperatura_promedio_dia'], marker='o', linestyle='-')
# plt.axhline(y=umbral_alta, color='r', linestyle='--', label='Umbral Alta')
# plt.axhline(y=umbral_baja, color='b', linestyle='--', label='Umbral Baja')
# plt.title('Temperatura Promedio Diario')
# plt.xlabel('Fecha')
# plt.ylabel('Temperatura (°C)')
# plt.legend()
# plt.grid()
# plt.show()


# guardo el nuevo dataset, esto es opcional, lo hago para verlo de forma visual
df_temp_subset.to_csv('./Dataset_INUMET/CSV/Datos_procesados/Temperatura_procesada.csv', sep=';', index=False)