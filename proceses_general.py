import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from datetime import datetime

# Datos de usuario y fecha
usuario = "facuguisande"  # Usando tu login proporcionado
fecha_actual = datetime.now().strftime("%Y%m%d")

print(f"=== Procesamiento de Datos Climáticos - {usuario} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

# 1. PROCESAMIENTO DE DATOS DE HUMEDAD
print("PASO 1: Procesando datos de humedad...")

try:
    df_humedad = pd.read_excel('./archivos/Datos.xlsx', sheet_name="HR", skiprows=1)
    print(f"Archivo de humedad cargado - Hoja 'HR' - Dimensiones: {df_humedad.shape}")
except Exception as e:
    print(f"Error al cargar la hoja 'HR': {e}")
    try:
        df_humedad = pd.read_excel('./archivos/Datos.xlsx', sheet_name=1, skiprows=1)
        print(f"Archivo de humedad cargado - Hoja #2 - Dimensiones: {df_humedad.shape}")
    except Exception as e2:
        print(f"Error crítico al cargar datos de humedad: {e2}")
        exit()

# Procesamiento de humedad (como antes)
df_humedad.columns = ['fecha_hora', 'humedad_ob1', 'humedad_ob2']
imputer = SimpleImputer(strategy='mean')
df_humedad[['humedad_ob1', 'humedad_ob2']] = imputer.fit_transform(df_humedad[['humedad_ob1', 'humedad_ob2']])
df_humedad['fecha'] = df_humedad['fecha_hora'].dt.date
umbral_humedad = 60
df_humedad['humedo_ob1'] = (df_humedad['humedad_ob1'] >= umbral_humedad).astype(int)
df_humedad['humedo_ob2'] = (df_humedad['humedad_ob2'] >= umbral_humedad).astype(int)
df_humedad_diaria = df_humedad.groupby('fecha').agg({
    'humedad_ob1': 'mean',
    'humedad_ob2': 'mean',
    'humedo_ob1': 'mean',
    'humedo_ob2': 'mean'
}).reset_index()
df_humedad_diaria['humedo_ob1'] = df_humedad_diaria['humedo_ob1'].round().astype(int)
df_humedad_diaria['humedo_ob2'] = df_humedad_diaria['humedo_ob2'].round().astype(int)

print(f"Datos de humedad procesados - {len(df_humedad_diaria)} días únicos")

# 2. PROCESAMIENTO DE DATOS DE TEMPERATURA
print("\nPASO 2: Procesando datos de temperatura...")

try:
    df_temperatura = pd.read_excel('./archivos/temperaturas_dia_prom_test.xlsx')
    print(f"Archivo de temperatura cargado - Dimensiones: {df_temperatura.shape}")
except Exception as e:
    print(f"Error al cargar datos de temperatura: {e}")
    print("Se continuará sin datos de temperatura")
    df_temperatura = None

if df_temperatura is not None:
    print("Columnas en el archivo de temperatura:")
    print(df_temperatura.columns.tolist())
    
    try:
        df_temperatura.columns = ['fecha', 'temp_ob1', 'temp_ob2']
    except ValueError:
        print("Error al renombrar columnas. El número de columnas no coincide.")
        print("Por favor verifica la estructura del archivo de temperatura y ajusta el código.")
        print(df_temperatura.head())
        exit()
    
    df_temperatura['fecha'] = pd.to_datetime(df_temperatura['fecha']).dt.date
    imputer_temp = SimpleImputer(strategy='mean')
    df_temperatura[['temp_ob1', 'temp_ob2']] = imputer_temp.fit_transform(df_temperatura[['temp_ob1', 'temp_ob2']])
    
    print(f"Datos de temperatura procesados - {len(df_temperatura)} registros")

# 3. COMBINAR DATOS DE HUMEDAD Y TEMPERATURA
print("\nPASO 3: Combinando datos de humedad y temperatura...")

# Si tenemos datos de temperatura, hacer el merge
if df_temperatura is not None:
    # Merge de los dataframes por fecha
    df_combinado = pd.merge(
        df_humedad_diaria,
        df_temperatura,
        on='fecha',
        how='outer'  # Usar outer join para mantener todas las fechas de ambos datasets
    )
    
    print(f"Datos combinados - {len(df_combinado)} registros totales")
    
    # Analizar fechas sin datos completos
    fechas_sin_humedad = df_combinado[df_combinado['humedad_ob1'].isna()].copy()
    fechas_sin_temp = df_combinado[df_combinado['temp_ob1'].isna()].copy()
    
    if len(fechas_sin_humedad) > 0:
        print(f"ADVERTENCIA: {len(fechas_sin_humedad)} días sin datos de humedad")
    if len(fechas_sin_temp) > 0:
        print(f"ADVERTENCIA: {len(fechas_sin_temp)} días sin datos de temperatura")
        
        # Crear un archivo separado con las fechas sin datos de temperatura
        archivo_fechas_incompletas = f'fechas_sin_temperatura_{usuario}_{fecha_actual}.xlsx'
        fechas_sin_temp.to_excel(archivo_fechas_incompletas, index=False)
        print(f"Se ha guardado un listado de fechas sin datos de temperatura en '{archivo_fechas_incompletas}'")
    
    # *** MANEJO DE VALORES FALTANTES ***
    # Método seleccionado: IMPUTACIÓN PERSONALIZADA
    
    print("\nAplicando tratamiento a valores faltantes de temperatura...")
    
    # 1. Cálculo de estadísticas mensuales para temperaturas
    df_combinado['mes'] = pd.to_datetime(df_combinado['fecha']).dt.month
    
    # Calcular medias mensuales para los datos existentes
    medias_mensuales = df_combinado.groupby('mes')[['temp_ob1', 'temp_ob2']].mean()
    print("Temperaturas medias mensuales calculadas para imputación:")
    print(medias_mensuales)
    
    # 2. Imputar valores faltantes de temperatura basados en la media del mes correspondiente
    for index, row in df_combinado[df_combinado['temp_ob1'].isna()].iterrows():
        mes = row['mes']
        # Si existe información para ese mes, usar la media mensual
        if mes in medias_mensuales.index:
            df_combinado.loc[index, 'temp_ob1'] = medias_mensuales.loc[mes, 'temp_ob1']
            df_combinado.loc[index, 'temp_ob2'] = medias_mensuales.loc[mes, 'temp_ob2']
            print(f"Fecha {row['fecha']}: Temperatura imputada con media del mes {mes}")
        # Si no hay información para ese mes, usar la media general
        else:
            df_combinado.loc[index, 'temp_ob1'] = df_combinado['temp_ob1'].mean()
            df_combinado.loc[index, 'temp_ob2'] = df_combinado['temp_ob2'].mean()
            print(f"Fecha {row['fecha']}: Temperatura imputada con media general (no hay datos para el mes {mes})")
    
    # Eliminar columna auxiliar de mes
    df_combinado.drop(columns=['mes'], inplace=True)
    
    # 3. Imputar cualquier otro valor faltante con la media general (caso extremo)
    for col in ['humedad_ob1', 'humedad_ob2', 'temp_ob1', 'temp_ob2']:
        if df_combinado[col].isna().any():
            media = df_combinado[col].mean()
            df_combinado[col].fillna(media, inplace=True)
    
    # Recalcular las clasificaciones binarias
    df_combinado['humedo_ob1'] = (df_combinado['humedad_ob1'] >= umbral_humedad).astype(int)
    df_combinado['humedo_ob2'] = (df_combinado['humedad_ob2'] >= umbral_humedad).astype(int)
    
    # 4. Añadir columna de confiabilidad/fuente de datos
    df_combinado['temp_es_imputada'] = 0  # Por defecto, asumimos que no es imputada
    for fecha in fechas_sin_temp['fecha'].values:
        df_combinado.loc[df_combinado['fecha'] == fecha, 'temp_es_imputada'] = 1
        
else:
    # Si no hay datos de temperatura, usar solo los datos de humedad
    df_combinado = df_humedad_diaria
    # Añadir columnas de temperatura con valores NaN
    df_combinado['temp_ob1'] = np.nan
    df_combinado['temp_ob2'] = np.nan
    df_combinado['temp_es_imputada'] = 1  # Todos los valores serán imputados
    print("No hay datos de temperatura disponibles. Se han agregado columnas de temperatura con valores nulos.")

# 4. GUARDAR RESULTADOS
print("\nPASO 4: Guardando resultados...")

# Ordenar por fecha
df_combinado = df_combinado.sort_values('fecha')

# Guardar los dataframes combinados
archivo_salida = f'clima_combinado_{usuario}_{fecha_actual}.xlsx'

try:
    # Crear un ExcelWriter
    with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
        # Guardar los datos combinados en la primera hoja
        df_combinado.to_excel(writer, sheet_name='Datos_Combinados', index=False)
        
        # Crear una segunda hoja con datos no imputados (solo registros originales)
        if 'temp_es_imputada' in df_combinado.columns:
            datos_originales = df_combinado[df_combinado['temp_es_imputada'] == 0].drop(columns=['temp_es_imputada'])
            datos_originales.to_excel(writer, sheet_name='Solo_Datos_Originales', index=False)
            
            # Crear una tercera hoja con datos imputados
            datos_imputados = df_combinado[df_combinado['temp_es_imputada'] == 1].drop(columns=['temp_es_imputada'])
            datos_imputados.to_excel(writer, sheet_name='Solo_Datos_Imputados', index=False)
    
    print(f"Archivo '{archivo_salida}' creado exitosamente con {len(df_combinado)} registros totales")
    if 'temp_es_imputada' in df_combinado.columns:
        originales = (df_combinado['temp_es_imputada'] == 0).sum()
        imputados = (df_combinado['temp_es_imputada'] == 1).sum()
        print(f"  - {originales} registros con datos completos originales")
        print(f"  - {imputados} registros con temperatura imputada")
    
    # Mostrar las primeras filas del resultado
    print("\nPrimeras filas del archivo combinado:")
    print(df_combinado.head())
except Exception as e:
    print(f"Error al guardar el archivo: {e}")

# 5. GENERAR INFORME ESTADÍSTICO
print("\nPASO 5: Generando informe estadístico...")

# Crear un DataFrame de estadísticas
estadisticas = []

# Estadísticas generales
estadisticas.append(["Total de días procesados", len(df_combinado)])
estadisticas.append(["Período analizado", f"{df_combinado['fecha'].min()} al {df_combinado['fecha'].max()}"])

# Estadísticas de completitud de datos
if 'temp_es_imputada' in df_combinado.columns:
    registros_completos = (df_combinado['temp_es_imputada'] == 0).sum()
    registros_imputados = (df_combinado['temp_es_imputada'] == 1).sum()
    estadisticas.append(["Días con datos completos", f"{registros_completos} ({registros_completos/len(df_combinado)*100:.1f}%)"])
    estadisticas.append(["Días con temperatura imputada", f"{registros_imputados} ({registros_imputados/len(df_combinado)*100:.1f}%)"])

# Estadísticas de humedad
humedo_ob1_count = df_combinado['humedo_ob1'].sum()
humedo_ob2_count = df_combinado['humedo_ob2'].sum()
estadisticas.append(["Días húmedos (OB1)", f"{humedo_ob1_count} ({humedo_ob1_count/len(df_combinado)*100:.1f}%)"])
estadisticas.append(["Días húmedos (OB2)", f"{humedo_ob2_count} ({humedo_ob2_count/len(df_combinado)*100:.1f}%)"])

# Estadísticas de temperatura (solo para registros no imputados)
if 'temp_ob1' in df_combinado.columns:
    # Estadísticas generales
    temp_ob1_avg = df_combinado['temp_ob1'].mean()
    temp_ob2_avg = df_combinado['temp_ob2'].mean()
    estadisticas.append(["Temperatura promedio general (OB1)", f"{temp_ob1_avg:.1f}°C"])
    estadisticas.append(["Temperatura promedio general (OB2)", f"{temp_ob2_avg:.1f}°C"])
    
    # Estadísticas solo para datos no imputados
    if 'temp_es_imputada' in df_combinado.columns and registros_completos > 0:
        datos_originales = df_combinado[df_combinado['temp_es_imputada'] == 0]
        temp_ob1_avg_orig = datos_originales['temp_ob1'].mean()
        temp_ob2_avg_orig = datos_originales['temp_ob2'].mean()
        estadisticas.append(["Temperatura promedio (solo datos originales) (OB1)", f"{temp_ob1_avg_orig:.1f}°C"])
        estadisticas.append(["Temperatura promedio (solo datos originales) (OB2)", f"{temp_ob2_avg_orig:.1f}°C"])
    
    # Correlación entre temperatura y humedad (solo datos originales)
    if 'temp_es_imputada' in df_combinado.columns and registros_completos > 0:
        corr_temp1_hum1_orig = datos_originales['temp_ob1'].corr(datos_originales['humedad_ob1'])
        estadisticas.append(["Correlación Temp-Humedad (solo datos originales) (OB1)", f"{corr_temp1_hum1_orig:.2f}"])

# Discrepancias entre observatorios
discrepancias = df_combinado[df_combinado['humedo_ob1'] != df_combinado['humedo_ob2']]
estadisticas.append(["Días con clasificación diferente de humedad", f"{len(discrepancias)} ({len(discrepancias)/len(df_combinado)*100:.1f}%)"])

# Crear DataFrame de estadísticas
df_stats = pd.DataFrame(estadisticas, columns=['Métrica', 'Valor'])

# Guardar estadísticas
archivo_stats = f'estadisticas_clima_{usuario}_{fecha_actual}.xlsx'
df_stats.to_excel(archivo_stats, index=False)
print(f"Informe estadístico guardado en '{archivo_stats}'")

print("\n=== Procesamiento finalizado ===")