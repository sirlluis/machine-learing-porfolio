-- Ventas acumuladas por mes
SELECT
    SUBSTR("Order Date", 7, 4) AS year,
    PRINTF('%02d', CAST(SUBSTR("Order Date", 1, INSTR("Order Date", '/') - 1) AS INTEGER)) AS month,
    ROUND(SUM(Sales), 2) AS monthly_sales,
    ROUND(SUM(SUM(Sales)) OVER (
        PARTITION BY SUBSTR("Order Date", 7, 4)
        ORDER BY PRINTF('%02d', CAST(SUBSTR("Order Date", 1, INSTR("Order Date", '/') - 1) AS INTEGER))
    ), 2) AS cumulative_sales
FROM ventas
GROUP BY year, month
ORDER BY year, month;