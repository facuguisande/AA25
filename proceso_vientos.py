import pandas as pd

archivo_vientos = './Dataset_INUMET/CSV/VientoCarrasco.csv'

try:
    df_vientos = pd.read_csv(archivo_vientos , sep=';')
    print("DataFrame cargado con éxito desde el archivo CSV")
except FileNotFoundError:
    print(f"Error: El archivo '{archivo_vientos}' no se encontró.")
    exit()
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")
    exit()

# Asumo que la primera columna es la fecha.
# Aca renombro para que me quede mas facil ubicar y no por el nombre largo.
if df_vientos.columns[0] is not None:
     df_vientos[df_vientos.columns[0]] = pd.to_datetime(df_vientos[df_vientos.columns[0]], dayfirst=True)
     df_vientos.rename(columns={df_vientos.columns[0]: 'Fecha'}, inplace=True)
else:
    print("Advertencia: La primera columna no tiene nombre y no se pudo renombrar a 'Fecha'.")
    df_vientos.columns.values[0] = 'Fecha' # Asignar nombre directamente
    df_vientos['Fecha'] = pd.to_datetime(df_vientos['Fecha'], dayfirst=True)

# me quedo solo con la primera y segunda columna, serian los datos de carrasco
df_vientos = df_vientos.iloc[:, :2].copy()

print(df_vientos.head())
print("Verificando valores nulos en el DataFrame:")
print(df_vientos.isnull().sum())

df_vientos.rename(columns={df_vientos.columns[1]: 'DirVientos'}, inplace=True)

print("DataFrame original:")
print(df_vientos.head())

# Extraer solo la fecha (sin hora) para agrupar por día, como los demas dataframes que tengo
df_vientos['FechaSolo'] = df_vientos['Fecha'].dt.date

# Calcular la moda de la dirección del viento por día
moda_por_dia = df_vientos.groupby('FechaSolo')['DirVientos'].apply(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
moda_por_dia.rename(columns={'FechaSolo': 'Fecha'}, inplace=True)

# print("\nModa de dirección del viento por día:")
# print(moda_por_dia.head())
# print(f"\nTotal de días procesados: {len(moda_por_dia)}")

df_vientos_moda_procesada = moda_por_dia.copy()

# Convertir la columna DirVientos a numérico, convirtiendo errores a NaN, ya que hay str en la columna
df_vientos_moda_procesada['DirVientos'] = pd.to_numeric(df_vientos_moda_procesada['DirVientos'], errors='coerce')

# convertir los grados a cardinales
def grados_a_cardinal(grados):
    if grados is None or pd.isna(grados):
        return 'Desconocido'
    try:
        grados = float(grados)
    except (ValueError, TypeError):
        return 'Desconocido'
    
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ix = round(grados / 45) % 8
    return dirs[ix] 

# Aplicar la conversión a la columna 'DirVientos' y crear una nueva columna 'DirVientos_Cardinal'
df_vientos_moda_procesada['DirVientos_Cardinal'] = df_vientos_moda_procesada['DirVientos'].apply(grados_a_cardinal)
# print(df_vientos_moda_procesada.head())
# print(f"\nValores nulos en el DataFrame final: {df_vientos_moda_procesada.isnull().sum()}")

# ubicación del valor nulo y lo elimino, aca primero printe y era uno solo, por lo cual eliminar una fila me parecio que no implicaba rangos
for col in df_vientos_moda_procesada.columns:
    if df_vientos_moda_procesada[col].isnull().any():
        print(f"Valores nulos encontrados en la columna '{col}':")
        print(df_vientos_moda_procesada[df_vientos_moda_procesada[col].isnull()])
        df_vientos_moda_procesada = df_vientos_moda_procesada[df_vientos_moda_procesada[col].notnull()]
        print(f"Después de eliminar, total de filas: {len(df_vientos_moda_procesada)}")
# print(f"\nValores nulos después de la limpieza: {df_vientos_moda_procesada.isnull().sum()}")
print(df_vientos_moda_procesada.head())

# guardar el DataFrame final a un nuevo archivo CSV, para verlo mejor de manera visual.
df_vientos_moda_procesada.to_csv('./Dataset_INUMET/CSV/Datos_procesados/Vientos_Procesado.csv', sep=';', index=False)

