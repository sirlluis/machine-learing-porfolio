-- Ventas y profit por región y segmento
SELECT
    Region,
    Segment,
    ROUND(SUM(Sales),2) AS total_sales,
    ROUND(SUM(Profit), 2) AS total_profit,
    ROUND(AVG(Discount)*100, 1) AS avg_discount -- porcentaje
FROM ventas
GROUP BY Region, Segment
ORDER BY Region, total_sales DESC;