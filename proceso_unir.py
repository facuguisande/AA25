import pandas as pd
import ephem

csv_temperatura = './Dataset_INUMET/CSV/Datos_procesados/Temperatura_procesada.csv'
csv_truenos = './Dataset_INUMET/CSV/Datos_procesados/Truenos_Procesado.csv'
csv_vientos = './Dataset_INUMET/CSV/Datos_procesados/Vientos_Procesado.csv'
csv_lluvias = './Dataset_INUMET/CSV/Datos_procesados/Lluvia_procesado.csv'

# Nombres de tus archivos CSV
csv_files = [csv_temperatura, csv_truenos, csv_vientos, csv_lluvias]

# Lista para almacenar los DataFrames individuales
dfs = []

# Cargar y pre-procesar cada CSV
for file in csv_files:
    print(f"\nProcesando archivo: {file}")

    df = pd.read_csv(file, sep=';')  # Usar punto y coma como separador
    
    # Asume que la primera columna es la fecha y la nombra 'Fecha'
    df.columns.values[0] = 'Fecha'
    
    # Convierte la columna 'Fecha' a tipo datetime (si no lo está ya)
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    # Eliminar filas con fechas inválidas
    df = df.dropna(subset=['Fecha'])
    
    # Eliminar duplicados por fecha, manteniendo el primer registro
    # df = df.drop_duplicates(subset=['Fecha'], keep='first')
    
    print(f"Después del procesamiento - Columnas: {list(df.columns)}")
    print(f"Fechas únicas: {len(df)}")
    
    dfs.append(df)

# Unir todos los DataFrames usando merge en la columna 'Fecha'
# Empezar con el primer DataFrame
df_final = dfs[0]
print(f"\nIniciando merge con DataFrame base: {df_final.shape}")

for i in range(1, len(dfs)):
    print(f"Mergeando con DataFrame {i+1}: {dfs[i].shape}")
    df_final = pd.merge(df_final, dfs[i], on='Fecha', how='outer')
    print(f"Resultado después del merge: {df_final.shape}")

# Ordenar por fecha
df_final = df_final.sort_values('Fecha').reset_index(drop=True)

# Muestra información detallada del DataFrame final
print("=== INFORMACIÓN DEL DATAFRAME UNIDO ===")
print(f"Forma del DataFrame: {df_final.shape}")
print(f"Número de filas: {df_final.shape[0]}")
print(f"Número de columnas: {df_final.shape[1]}")

print("\n=== NOMBRES DE TODAS LAS COLUMNAS ===")
for i, col in enumerate(df_final.columns):
    print(f"{i+1}. {col}")

# print("\n=== TIPOS DE DATOS ===")
# print(df_final.dtypes)

# print("\n=== PRIMERAS 10 FILAS ===")
# print(df_final.head(10))

# print("\n=== ÚLTIMAS 5 FILAS ===")
# print(df_final.tail(5))

# print("\n=== RANGO DE FECHAS ===")
# print(f"Fecha más antigua: {df_final['Fecha'].min()}")
# print(f"Fecha más reciente: {df_final['Fecha'].max()}")

# Verifica si hay valores NaN
print("\n=== VALORES NaN POR COLUMNA ===")
nan_counts = df_final.isnull().sum()
total_rows = len(df_final)
for col, count in nan_counts.items():
    percentage = (count / total_rows) * 100
    print(f"{col}: {count} valores nulos ({percentage:.1f}%)")

print(f"\n=== RESUMEN DE VALORES NULOS ===")
print(f"Total de filas: {total_rows}")
print(f"Total de valores nulos: {df_final.isnull().sum().sum()}")
print(f"Porcentaje de completitud: {((df_final.size - df_final.isnull().sum().sum()) / df_final.size * 100):.2f}%")

# # Guardar el DataFrame unido en un nuevo CSV
output_file = './Dataset_INUMET/CSV/Datos_procesados/Datos_Unidos_Procesados.csv'

# estaciones del mes, funcion para verificar.
def mes_a_estacion(mes):
    if mes in [12, 1, 2]:
        return 'Verano'  # Diciembre, Enero, Febrero
    elif mes in [3, 4, 5]:
        return 'Otoño'   # Marzo, Abril, Mayo
    elif mes in [6, 7, 8]:
        return 'Invierno' # Junio, Julio, Agosto
    elif mes in [9, 10, 11]:
        return 'Primavera' # Septiembre, Octubre, Noviembre
    else:
        return 'Desconocido'
# Aplico la función para crear la nueva columna 'Estacion'
df_final['Estacion'] = df_final['Fecha'].dt.month.apply(mes_a_estacion)
print("\n=== PRIMERAS 10 FILAS CON COLUMNA 'Estacion' ===")
print(df_final[['Fecha', 'Estacion']].head(10))

# Fase lunar, usando ephem
def fase_luna(fecha):
    luna = ephem.Moon(fecha)
    fase = luna.phase  # La fase de la luna en grados (0 a 360)
    
    # Determinación de la fase
    if fase < 7.4:
        return "Luna Nueva"
    elif fase < 14.8:
        return "Creciente"
    elif fase < 22.1:
        return "Luna Llena"
    else:
        return "Menguante"
df_final['Fase_Luna'] = df_final['Fecha'].apply(fase_luna)
print("\n=== PRIMERAS 10 FILAS CON COLUMNA 'Fase_Luna' ===")
print(df_final[['Fecha', 'Fase_Luna']].head(10))

#guardo nuevamente el archivo con la nueva columna
try:
    df_final.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"\n=== GUARDADO EXITOSO CON COLUMNA 'Estacion' ===")
    print(f"Archivo guardado en: {output_file}")
    print("Separador usado: punto y coma (;) - Compatible con Excel")
except Exception as e:
    print(f"\n=== ERROR AL GUARDAR ===")
    print(f"Error: {e}")
    print("Intentando guardar en la carpeta actual...")
   

