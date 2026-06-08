-- Ranking de productos por profit
SELECT
    "Product Name",
    Category,
    ROUND(SUM(Profit), 2) AS total_profit,
    RANK() OVER (PARTITION BY Category ORDER BY SUM(Profit) DESC) AS rank_in_category
FROM ventas
GROUP BY "Product Name", Category
ORDER BY Category, rank_in_category
LIMIT 20;