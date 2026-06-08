-- Subcategorías con pérdidas
SELECT "Sub-Category",
    ROUND(SUM(Profit),2) AS total_profit
FROM ventas
GROUP BY "Sub-Category"
HAVING total_profit<0;