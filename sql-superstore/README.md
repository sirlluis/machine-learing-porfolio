# 🛒 Superstore Sales Analysis — SQL + Power BI

A data analysis project exploring 4 years of retail sales data using **SQL** for querying and aggregation, and **Power BI** for interactive dashboard visualization.

---

## 📌 Project Overview

This project demonstrates an end-to-end analytics workflow applied to the [Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) — a widely-used retail dataset covering orders, customers, products, and profitability across the United States.

**Key Questions Explored:**
- Which product categories and sub-categories generate the most — and least — profit?
- Who are the top customers by revenue and order frequency?
- How do sales and profitability vary across regions and customer segments?
- Which sub-categories are operating at a loss, and why?
- What does cumulative monthly revenue look like year over year?

---

## 📂 Project Structure

```
superstore-sql/
│
├── data/
│   └── Sample-Superstore.csv          # Raw dataset (source: Kaggle)
│
├── scripts/
│   └── load_data.py                   # CSV → SQLite pipeline
│
├── queries/
│   ├── 01_ventas_por_categoria.sql    # Total sales & profit by category
│   ├── 02_top_clientes.sql            # Top 10 customers by revenue
│   ├── 03_region_segmento.sql         # Sales breakdown by region & segment
│   ├── 04_subcategorias_perdidas.sql  # Sub-categories with negative profit
│   ├── 05_ranking_productos.sql       # Product ranking with RANK() per category
│   └── 06_ventas_acumuladas.sql       # Monthly cumulative sales (window function)
│
└── README.md
```

---

## 🗄️ Dataset

| Field | Detail |
|---|---|
| **Source** | Kaggle — Superstore Dataset |
| **Records** | 9,994 rows |
| **Time range** | 4 years of orders |
| **Key columns** | Order Date, Ship Mode, Segment, Region, Category, Sub-Category, Sales, Quantity, Discount, Profit |
| **Missing values** | None |

---

## ⚙️ Workflow

### 1. Data Ingestion
The raw CSV is loaded into a local **SQLite** database using a Python script (`load_data.py`). SQLite was chosen for portability — no server required, and Power BI connects to it natively.

```bash
python scripts/load_data.py
# Output: superstore.db (SQLite database)
```

### 2. SQL Analysis
Queries are organized by complexity in the `queries/` folder:

| File | Technique |
|---|---|
| `01_ventas_por_categoria.sql` | `GROUP BY`, `SUM`, `ROUND` |
| `02_top_clientes.sql` | `COUNT(DISTINCT)`, `ORDER BY`, `LIMIT` |
| `03_region_segmento.sql` | Multi-column `GROUP BY`, `AVG` |
| `04_subcategorias_perdidas.sql` | `HAVING`, filtering on aggregates |
| `05_ranking_productos.sql` | `RANK()` window function, `PARTITION BY` |
| `06_ventas_acumuladas.sql` | `SUM() OVER()` cumulative window function |

### 3. Dashboard
Power BI connects directly to `superstore.db` and visualizes the query results as an interactive dashboard.

---

## 📊 Key Findings

- **Technology** is the top revenue category; **Office Supplies** leads in order volume
- **Tables** and **Bookcases** sub-categories operate at a net loss despite positive sales
- The **West** region generates the highest total profit; **Central** underperforms
- **Consumer** segment accounts for the majority of orders across all regions
- Revenue shows consistent year-over-year growth with Q4 peaks

---

## 🛠️ Technologies & Tools

- **Languages:** Python 3.10+, SQL
- **Database:** SQLite
- **Data Processing:** Pandas
- **Visualization:** Power BI
- **SQL Client:** DataGrip
- **Version Control:** Git & GitHub
- **Data Source:** [Kaggle — Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/sirlluis/superstore-sql.git
cd superstore-sql
```

**2. Install dependencies**
```bash
pip install pandas
```

**3. Generate the database**
```bash
python scripts/load_data.py
```

**4. Explore the queries**
Open any `.sql` file in DB Browser for SQLite and run against `superstore.db`

**5. Open the dashboard**
Open the `.pbix` file in Power BI Desktop

---

## 📬 Contact & Connect

- 📧 **Email:** luisgra@tuta.io
- 💼 **LinkedIn:** https://www.linkedin.com/in/lgra1525/
- 🐱 **GitHub:** https://github.com/sirlluis
- 📊 **Kaggle:** https://www.kaggle.com/luisgerardoram

---

## 📄 License

This project is licensed under the MIT License.

---

**Last Updated:** June 2026
**Status:** 🚀 In progress — dashboard in development