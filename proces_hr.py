import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from datetime import datetime

# 1. Cargar el archivo Excel - específicamente la hoja "HR" (hoja 2)
try:
    # Especificamos sheet_name="HR" para cargar la hoja correcta
    df = pd.read_excel('./archivos/Datos.xlsx', sheet_name="HR", skiprows=1)
    print("Archivo cargado exitosamente - Hoja 'HR'")
    print(f"Dimensiones del dataset: {df.shape}")
except Exception as e:
    print(f"Error al cargar la hoja 'HR': {e}")
    # Intento alternativo usando el índice de la hoja
    try:
        df = pd.read_excel('./archivos/Datos.xlsx', sheet_name=1, skiprows=1)
        print("Archivo cargado exitosamente - Hoja #2")
        print(f"Dimensiones del dataset: {df.shape}")
    except Exception as e2:
        print(f"Error al cargar la hoja por índice: {e2}")
        exit()

# 2. Mostrar las primeras filas para entender la estructura
print("\nPrimeras filas del dataset:")
print(df.head())

# 3. Renombrar columnas para trabajar más fácilmente
# Asumimos que las columnas son: fecha/hora, humedad_ob1, humedad_ob2
columnas_originales = df.columns.tolist()
print(f"Columnas originales: {columnas_originales}")
df.columns = ['fecha_hora', 'humedad_ob1', 'humedad_ob2']

# 4. Verificar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())
print(f"Total de filas: {len(df)}")
print(f"Filas con valores nulos: {df.isnull().any(axis=1).sum()}")

# 5. Tratar valores nulos usando SimpleImputer
# Usamos la media para imputar los valores faltantes de humedad
imputer = SimpleImputer(strategy='mean')
df[['humedad_ob1', 'humedad_ob2']] = imputer.fit_transform(df[['humedad_ob1', 'humedad_ob2']])

# 6. Extraer solo la fecha (sin hora)
df['fecha'] = df['fecha_hora'].dt.date

# 7. Definir umbral para considerar un día como húmedo
# Consideramos un día húmedo si la humedad relativa es >= 60%
umbral_humedad = 60
print(f"\nUmbral para clasificar día como húmedo: {umbral_humedad}%")

# 8. Clasificar días húmedos (1) y no húmedos (0)
df['humedo_ob1'] = (df['humedad_ob1'] >= umbral_humedad).astype(int)
df['humedo_ob2'] = (df['humedad_ob2'] >= umbral_humedad).astype(int)

# 9. Crear un nuevo DataFrame con solo los datos que necesitamos
nuevo_df = df[['fecha', 'humedo_ob1', 'humedo_ob2']].copy()

# 10. Si hay múltiples entradas para la misma fecha, tomar la media y redondear
nuevo_df = nuevo_df.groupby('fecha').mean().round().reset_index()

# 11. Guardar resultado en un nuevo archivo Excel
fecha_actual = datetime.now().strftime("%Y%m%d")
usuario = "facuguisande"  # Tu nombre de usuario
nuevo_archivo = f'clasificacion_humedad_{usuario}_{fecha_actual}.xlsx'

try:
    nuevo_df.to_excel(nuevo_archivo, index=False)
    print(f"\nArchivo '{nuevo_archivo}' creado exitosamente")
    print("\nPrimeras filas del nuevo dataset:")
    print(nuevo_df.head())
except Exception as e:
    print(f"Error al guardar el archivo: {e}")

# 12. Mostrar estadísticas sobre los días húmedos
print("\nEstadísticas de días húmedos:")
print(f"Observatorio 1: {nuevo_df['humedo_ob1'].sum()} días húmedos de {len(nuevo_df)} ({nuevo_df['humedo_ob1'].mean()*100:.1f}%)")
print(f"Observatorio 2: {nuevo_df['humedo_ob2'].sum()} días húmedos de {len(nuevo_df)} ({nuevo_df['humedo_ob2'].mean()*100:.1f}%)")

# 13. Verificar si hay discrepancias entre los observatorios
discrepancias = nuevo_df[nuevo_df['humedo_ob1'] != nuevo_df['humedo_ob2']]
print(f"\nDías con clasificación diferente entre observatorios: {len(discrepancias)} ({len(discrepancias)/len(nuevo_df)*100:.1f}%)")
if not discrepancias.empty:
    print("\nPrimeros ejemplos de días con discrepancias:")
    print(discrepancias.head())

# 14. Generar un segundo archivo con información adicional (opcional)
informe_df = pd.DataFrame({
    'Métrica': [
        'Total de días analizados',
        'Días húmedos OB1',
        'Días húmedos OB2',
        'Días con discrepancias',
        'Fecha de procesamiento',
        'Umbral de humedad utilizado'
    ],
    'Valor': [
        len(nuevo_df),
        nuevo_df['humedo_ob1'].sum(),
        nuevo_df['humedo_ob2'].sum(),
        len(discrepancias),
        datetime.now().strftime("%Y-%m-%d"),
        f"{umbral_humedad}%"
    ]
})

informe_archivo = f'informe_humedad_{usuario}_{fecha_actual}.xlsx'
informe_df.to_excel(informe_archivo, index=False)
print(f"\nInforme resumen guardado en '{informe_archivo}'")