SELECT
    "Customer Name",
    ROUND(SUM(Sales), 2) as total_sales,
    COUNT(DISTINCT "Order ID") AS total_orders
FROM ventas
GROUP BY "Customer Name"
ORDER BY total_sales DESC
LIMIT 10;