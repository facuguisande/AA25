Es un tipeo general de como se podria llegar a realizar el procesamiento.

En los CSV de lluvia y temperatura, trato los valores nulos con la media, pero quizas podamos usar alguna otra cosita, o alguna herramienta de sklearn.

Luego que hagamos el merge, van a reslutar valores Nan tambien, ya que pued que en una fecha no tengamos datos, de un determinado atributo. Aca tambien vamos a tener que definir una estrategia de tratemiento.
-Imputacion
-Moda, mediana o algo de eso
-Sino usar alguna herramienta de sklearn, que tenga ML para los datos

- En la ruta Dataset_Inumet/CSV/Datos procesados, estan las salidas de proceso_lluvia y proceso_temp
Los pueden elimianr y generar de nuevo, hay algunos print que comenté, como los hice en visual y no en un jupyter, fui ejecutando y armando el codigo por partes, me ayudo a entender. Pueden ir comentando hasta ver los valores nulos, e ir descmoentando a medida que avanzan. 
O se lo pueden tirar a una IA y que les de una mano jaja