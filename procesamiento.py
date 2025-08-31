import pandas as pd

# Define la ruta de tu archivo Excel
archivo_excel = './archivos/Datos.xlsx' # ¡Asegúrate de cambiar esto por el nombre de tu archivo!

# Nombre de la hoja en la que están tus datos de temperatura
nombre_hoja = 'TmaxTmin' # ¡Asegúrate de cambiar esto si tu hoja tiene otro nombre!

# 1. Cargar la hoja específica del archivo Excel
try:
    df = pd.read_excel(archivo_excel, sheet_name=nombre_hoja)
    print("DataFrame cargado con éxito desde la hoja:", nombre_hoja)
    print(df.head())
except FileNotFoundError:
    print(f"Error: El archivo '{archivo_excel}' no se encontró.")
    exit()
except Exception as e:
    print(f"Error al leer el archivo Excel o la hoja '{nombre_hoja}': {e}")
    exit()

# Asegurarse de que la primera columna sea de tipo fecha y renombrarla a 'Fecha'
# Asumo que la primera columna es la fecha.
if df.columns[0] is not None:
     df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
     df.rename(columns={df.columns[0]: 'Fecha'}, inplace=True) # Renombrar para mayor claridad
else:
    print("Advertencia: La primera columna no tiene nombre y no se pudo renombrar a 'Fecha'.")
    df.columns.values[0] = 'Fecha' # Asignar nombre directamente
    df['Fecha'] = pd.to_datetime(df['Fecha'])


# Renombrar las columnas de temperatura para facilitar el acceso
# Asumo que después de 'Fecha', las siguientes 4 columnas son las temperaturas:
# Tmax_Obs1, Tmin_Obs1, Tmax_Obs2, Tmin_Obs2
if len(df.columns) >= 5: # Contando 'Fecha' y las 4 temperaturas
    # Nombres de ejemplo, ajústalos a los nombres reales de tus columnas si los tienen
    # o usa df.columns[1], df.columns[2], etc.
    df.rename(columns={df.columns[1]: 'Tmax_Obs1',
                       df.columns[2]: 'Tmin_Obs1',
                       df.columns[3]: 'Tmax_Obs2',
                       df.columns[4]: 'Tmin_Obs2'}, inplace=True)
else:
    print("Advertencia: No se encontraron las 4 columnas de temperatura esperadas. Asegúrate de que tu hoja tenga al menos 5 columnas de datos (Fecha + 4 temperaturas).")
    exit()

# 2. Calcular la temperatura promedio diaria para cada observatorio directamente
# Cada fila ya es un día, así que simplemente promediamos Tmax y Tmin para cada observatorio.

# Temperatura promedio del día para Observatorio 1
df['Temp_Promedio_Obs1'] = (df['Tmax_Obs1'] + df['Tmin_Obs1']) / 2

# Temperatura promedio del día para Observatorio 2
df['Temp_Promedio_Obs2'] = (df['Tmax_Obs2'] + df['Tmin_Obs2']) / 2

# 3. Calcular el promedio general de temperatura entre ambos observatorios por día
df['Temp_Promedio_Global_Dia'] = (df['Temp_Promedio_Obs1'] + df['Temp_Promedio_Obs2']) / 2

print("\nDataFrame con las temperaturas promedio diarias calculadas:")
print(df.head())

# 4. Seleccionar las columnas relevantes para el nuevo archivo
# Queremos 'Fecha', 'Temp_Promedio_Obs1', 'Temp_Promedio_Obs2', 'Temp_Promedio_Global_Dia'
df_resultado = df[['Fecha', 'Temp_Promedio_Obs1', 'Temp_Promedio_Obs2', 'Temp_Promedio_Global_Dia']].copy()

# 5. Crear un nuevo archivo CSV con los resultados
nombre_salida = './archivos/temperaturas_diarias_promedio.csv'
df_resultado.to_csv(nombre_salida, index=False)

print(f"\nDatos de temperatura promedio diaria guardados en '{nombre_salida}'")
