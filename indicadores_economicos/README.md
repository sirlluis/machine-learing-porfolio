# Indicadores económicos
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-purple?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-datos-150458?style=flat-square&logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/NumPy-cálculo-013243?style=flat-square&logo=numpy&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-viz-pink?style=flat-square)
![seaborn](https://img.shields.io/badge/seaborn-viz-c44e52?style=flat-square)
![PCA](https://img.shields.io/badge/Técnica-PCA-7c3aed?style=flat-square)
![KMeans](https://img.shields.io/badge/Clustering-K--Means-2e7d32?style=flat-square)
![Agglomerative](https://img.shields.io/badge/Clustering-Agglomerative-2e7d32?style=flat-square)
![DBSCAN](https://img.shields.io/badge/Clustering-DBSCAN-2e7d32?style=flat-square)
![AffinityPropagation](https://img.shields.io/badge/Clustering-Affinity%20Propagation-2e7d32?style=flat-square)
![GaussianMixture](https://img.shields.io/badge/Clustering-Gaussian%20Mixture-2e7d32?style=flat-square)
![BIRCH](https://img.shields.io/badge/Clustering-BIRCH-2e7d32?style=flat-square)

# Descripción

Este proyecto tiene como objetivos reducir la información de 11 atributos económicos y sociales en unas nuevas variables que engloben todos los indicadores y tratar de agrupar a los países en clusters lo más definidos posibles para tener una mejor estructura económica y social del mundo.

Para esta actividad se consideran los siguientes predictores:

| Variable | Descripción |
|----------|-------------|
| Tasa | Tasa anual de crecimiento de la población. |
| Mortalidad | Tasa de mortalidad infantil por cada 1000 nacidos vivos. |
| Mujeres | Porcentaje de mujeres en la población activa. |
| PNB | Producto Nacional Bruto (en millones de dólares). |
| Luz | Producción de electricidad (en millones de KW/h). |
| Telefonía | Líneas telefónicas por cada 1000 habitantes. |
| Agua | Consumo de agua per cápita. |
| Bosques | Proporción de la superficie del país cubierta por bosques. |
| Deforestación | Proporción de deforestación anual. |
| Energía | Consumo de energía per cápita. |
| CO2 | Emisión de CO₂ per cápita. |

Se aplicará el método de Componentes Principales (PCA) con el objetivo de elegir el número óptimo de componentes y explicar qué pueden representar estos componentes respecto a los 11 indicadores. Estas componentes servirán para agrupar a los países en clusters mediante los siguientes algoritmos:
- `K-means`
- `Agglomerative clustering`
- `DBSCAN`
- `Affinity`
- `Gaussian mixtures`
- `BIRCH`

**Pregunta central:**

¿Qué se puede decir de México respecto al clúster al que pertenece?


# Conclusiones
Se probaron 6 modelos de clustering para clasificar los datos usando PCA, esto con dos objetivos:
- Reducir la dimensionalidad del conjunto de datos conservando la máxima varianza explicada.
- Reducir el impacto de valores atípicos y  el ruido.

Las componentes PCA obtenidas representan casi el 88% de la varianza explciada total, lo que parece razonable para la dimensión original de los daots (11 predictores).

Las dos primeras componentes de PCA  (PC1 y PC2) contienen la mayor parte de la varianza explicada, por lo que los gráficos de cada clsutering se llevaron a cabo tomando en cuenta dichos predictores.

## Elección del mejor modelo
Como hemos mencionado anteriormente, se probaron 6 algoritmos de clustering con las siguientes observaciones generales:
- `K-means`: identifica de una manera aceptable 5 clusters, dso principales etiquetados con 0 y 1, luego clasifica los 3 restantes como valores atípicos, cada uno. El gráfico `pairplot` muestra una separación relativamente buiena de lso datos procesados.
- `Agglomerative`: de una amnera similar a `K-means`, logra visualizar clusters separados, sin embargo, aquí se han obtenido 4 clusters, donde se pueden identificar correctamente los valores atípicos.
- `DBSCAN`: fue el de menor desempeño pues sobre ajustó los datos de tal manera que logró identificar dos clusters, el primero donde se encuentra la mayor densidad de los datos, y otro donde se muestran los valores atípicos, esto hace ver el poder de este algoritmo par aidentificar *outliers*.
- `Affinity`: a pesar de implementar técnicas para la obtención de los mejores hiperparámetros, el modelo identificaó 13 clusters, sin embargo, esta cantidad de clusters sobre ajusta el modelo y no es capaz de separar nítidamente los grupos.
- `GaussianMixture`: de los modelos vistos anetriormente, este es el que pudo clasificar de fomra exitosa 3 clsuters visiblemente separados, si bien existe aún un a mezcla de puntos, es notable diferencia respecto a los otros modelos, además de que los mejores hiperparámetros fueron encontrados iterando distintos modelos de covarianza, siendo `full`el elegido.
- `BIRCH`: de los modelos anteriores el más robusto para bases de datos muy grandes, se lograron resultados similares a los obtenidos con mezclas gaussianas, generadno 4 clusters relativamente bien definidos, destaca la identificación y clasificación de los valores atípicos. Este modelo fue probado para distintos valores de `threshold` para obtener el número óptimo de clusters y evitar el sobre ajuste.
---
En resumen, los modelos que clasificación de mejor forma los datos fueron `GaussianMixture`y `BIRCH`, por lo que se usarán para identificar los países en los datos clasificados.