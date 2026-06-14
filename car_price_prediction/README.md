# Predicción de precio de autos
# Sobre la base de datos
Este conjunto de datos contiene registros estructurados y orientados al consumidor de automóviles individuales que figuran en el mercado. Cada fila corresponde a un solo vehículo e incluye especificaciones técnicas, indicadores de uso y un precio de mercado. La colección está diseñada para ser clara, estar bien etiquetada y ser inmediatamente útil para el análisis exploratorio de datos, la ingeniería de características y el aprendizaje supervisado.Este conjunto de datos proporciona información detallada sobre varios modelos de automóviles y sus precios de mercado. Incluye características como la marca, el modelo, las especificaciones del motor, el tipo de combustible, el kilometraje y el tipo de transmisión, lo que lo hace ideal para la predicción de precios de automóviles, el análisis exploratorio de datos (EDA) y los proyectos de aprendizaje automático.

## Utilidad de esta base de datos
- Directamente accionable — las características son variables del mundo real comúnmente utilizadas en modelos de precios y valoración.
- 
- Compacto y enfocado — conjunto de características corto y de alta relevancia que reduce la carga de preprocesamiento.

- Versátil — admite tareas que van desde el análisis descriptivo y la visualización hasta la regresión (predicción de precios) y la clasificación (condición, tipo de combustible).

- Ideal para demostraciones y enseñanza — lo suficientemente pequeño para realizar iteraciones rápidas, pero lo bastante rico para permitir comparaciones significativas entre modelos.

## Descripción de las variables

| Atributo | Descripción |
|---|---|
| `Car ID` | Identificador único para cada vehículo |
| `Brand` | Fabricante (por ejemplo, Tesla, BMW, Toyota) |
| `Model` | Nombre específico del modelo |
| `Year` | Año de fabricación o modelo |
| `Engine Size` | Cilindrada del motor (en litros o cc) |
| `Fuel Type` | Tipo de combustible — Gasolina, Diésel, Eléctrico, Híbrido, etc. |
| `Transmission` | Tipo de transmisión — Manual o Automática |
| `Mileage` | Distancia total recorrida (en km o millas) |
| `Condition` | Estado categórico — Nuevo, Usado, Excelente, Regular |
| `Price` | Precio de venta o de mercado listado (variable objetivo) |