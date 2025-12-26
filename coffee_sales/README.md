# Predicción de ventas de café
Esta base de datos contiene los registros diarios de una máquina expendedora de café durante marzo de 2024 y parte de marzo de 2025.
## Objetivo
Existen distintas formas para analizar los datos, en este caso abordaremos lo siguiente:
- Análisis exploratorio de datos basados en series de tiempo
- Entrenar un modelo que permita predecir las ventas diarias
## Sobre la base de datos
El conjunto incluye registros de dos máquinas expendedoras de café nombrados `index_1.csv` e `index_2.csv`. Cada archivo contiene las siguientes columnas:
- `date`: fecha de la transacción
- `datetime`: fecha y hora de la transacción 
- `cash_type`: forma de pago
- `card`: identificador anónimo de la tarjeta
- `money`: monto pagado por transacción
- `coffee_name`: nombre de la bebida
## Métodología
El desarrollo del proyecto siguió las siguientes etapas:
- **Datos**: se importó la base de datos desde el archuvo `.csv`.
- **Preparción de los datos**:
  - Limpieza de datos nulos o faltantes.
  - Conversión de las columnas `date` y `datetime` al formato datetime de pandas para el manejo de fechas.
  - Extracción de datos temporales como año, mes, día y hora, en columnas separadas.
  - Codificación en variable *dummie* para la columna `coffee_name`.
  - Separación de predictores `X` y variable objetivo `money`.
  - División en datos de prueba y entrenamiento manteniendo el orden temporal, la proporción usada fue 80%-20%.
- **Análisis exloratorio**:
  - Promedio diario de ventas resaltando periódos útiles como verano e invierno para enriquecer el contexto.
  - Obtención del máximo y mínimo del ventas durante el periódo.
  - Bebida más vendida durante el periodo.
  - Bebida más vendida por mes.
  - Ventas mensuales por producto.
- **Implementación del modelo**

    Modelo de regresión lineal con los siguientes parámetros:
    - `fit_intercept`: True
    - `copy_X`: True
    - `tol`: 1e-06
    - `n_jobs`: None
    - `positive`: False
## Resultados
El modelo de regresión lineal obtuvo un error absoluto promedio (MAE) de aproximadamente **$0.4843$** lo cual indica un margen de error de menos la mita de un dolar por compra,lo que refleja en un desempeño bastante decente.
## Tecnologías utilizadas
- Python
- Scikit-learn
- Pandas
- VS Code
- Jupyter Notebooks
## Conclusiones
El modelo de regresión lineal puede capturar patrones relevantes relacionados a series de tiempo y ventas. Este modelo puede anticipar las ventas totales con una resolución de horas.

Dado el buen desempeño del  modeo, puede emplearse para hacer poryecciones y simulaciuones de venta de distintos productos y así anticipar gastos operativos, planear promociones o planificar días de baja venta.

LG