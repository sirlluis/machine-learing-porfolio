SELECT Category,
    ROUND(SUM(Sales), 2) as total_sales,
    ROUND(SUM(Profit), 2) as total_profit
FROM ventas
GROUP BY Category
ORDER BY total_sales DESC;