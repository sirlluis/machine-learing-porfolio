# 👋 Hi, I'm Luis Gerardo Ramírez Archundia

I am a **Physics graduate** from UAEMex with experience in **Quantum Chromodynamics (QCD)** and a strong interest in **Machine Learning and Data Science**.  
My projects combine knowledge of physics, mathematics, and programming to solve real-world problems through data-driven approaches.  

My goal is to apply my skills in **data analysis, statistical modeling, and machine learning** to innovative projects in the tech and scientific industries.  

---

# 📂 Machine Learning & Data Science Portfolio

This repository contains a comprehensive collection of projects showcasing my abilities in:  

- 📊 **Exploratory Data Analysis (EDA)** - Understanding data patterns and distributions
- 📈 **Time Series Forecasting** - ARIMA, LSTM, and machine learning models for temporal data
- 🎯 **Supervised Learning** - Regression and classification models  
- 🔍 **Unsupervised Learning** - Clustering (K-Means, DBSCAN, Agglomerative) and dimensionality reduction (PCA)
- 🧠 **Deep Learning** - Neural networks and computer vision applications
- ⚛️ **Physics + ML** - Applying machine learning techniques to physics-related problems
- 🗄️ **SQL & Database Analysis** - Data extraction, transformation, and insights from relational databases

Each project is documented with its own `README.md`, explaining the problem, methodology, and results.

---

## 🚀 Featured Projects

### 1. **SQL Superstore Analysis** 🗄️📊
**Directory:** `sql_superstore/`

Database analysis project showcasing SQL proficiency with comprehensive queries on a retail superstore dataset containing multiple years of sales data.

**Key Objectives:**
- Extract and analyze sales data using advanced SQL queries
- Perform multi-table joins to combine customer, order, and product information
- Generate insights on sales trends, regional performance, and product categories
- Calculate key metrics including revenue, profit, and customer behavior patterns

**Methodology:**
- Schema design with multiple related tables (Orders, Customers, Products, Shipping)
- Complex queries with GROUP BY, subqueries, and window functions
- Data aggregation by region, category, and time periods
- Performance optimization for large datasets

**Key Findings:**
- Regional sales distribution and profit margins
- Top performing product categories and segments
- Customer lifetime value and purchase patterns
- Seasonal trends and shipping method analysis

**Technologies:** SQL, MySQL/PostgreSQL, Database Design, Query Optimization

---

### 2. **Coffee Sales Prediction** ☕📊
**Directory:** `coffee_sales/`

A time series forecasting project analyzing daily coffee vending machine sales data spanning March 2024 - March 2025.

**Key Objectives:**
- Exploratory analysis of time series data with seasonal patterns
- Build predictive models for daily sales forecasting
- Extract temporal features (year, month, day, hour)

**Methodology:**
- Data preprocessing: handling missing values, datetime conversion
- Feature engineering: temporal feature extraction, one-hot encoding for coffee types
- Model: Linear Regression with train-test split (80%-20%) preserving temporal order

**Results:**
- MAE of $0.48 - indicating excellent prediction accuracy
- Successfully captures temporal patterns and sales trends
- Enables operational forecasting for inventory planning and promotion scheduling

**Technologies:** Python, Pandas, Scikit-learn, Matplotlib

---

### 3. **Coffee Shop Sales Analysis** ☕💰
**Directory:** `coffee_shop_sales/`

Comprehensive analysis of 149,116 transactions across three coffee shop locations (Lower Manhattan, Hell's Kitchen, Astoria) from January - June 2023.

**Key Questions:**
- Do the three locations behave similarly or are there differences?
- Which products generate the most revenue?
- How do customer preferences vary by location and time?

**Analysis Highlights:**

**Revenue Breakdown by Category:**
- ☕ Coffee: 38.6% of total revenue
- 🍵 Tea: 28.1%
- 🥐 Bakery: 11.8%
- Other: 21.5%

**Top Revenue Generators:**
1. Sustainably Grown Organic Lg (Hot Chocolate) - 3%+ of total sales
2. Dark Chocolate Lg
3. Latte Rg (Barista Espresso)

**Key Findings:**
- **Barista Espresso** dominates Coffee category (13.1% of total revenue)
- **Brewed Chai Tea** leads Tea category with strongest performance
- **Location Analysis:** Hell's Kitchen performs best for premium products; Astoria shows strongest Tea sales
- **Temporal Patterns:** Peak hours 7-10 AM; May-June are strongest months; Day 17 shows consistent spike

**Technologies:** Python, Pandas, Plotly, Seaborn, Matplotlib

---

### 4. **Economic Indicators Clustering** 🌍📊
**Directory:** `indicadores_economicos/`

Unsupervised learning project analyzing 11 economic and social indicators across 96 countries to identify country clusters and development patterns.

**Dataset Overview:**
- **96 countries** analyzed
- **11 indicators** tracked per country
- **No missing values**

**Indicators:**
| Variable | Description |
|----------|-------------|
| Tasa | Annual population growth rate |
| Mortalidad | Infant mortality per 1,000 live births |
| Mujeres | % of women in active workforce |
| PNB | Gross National Product (millions USD) |
| Luz | Electricity production (millions KWh) |
| Telefonía | Telephone lines per 1,000 inhabitants |
| Agua | Per capita water consumption |
| Bosques | Forest coverage % |
| Deforestación | Annual deforestation rate |
| Energía | Per capita energy consumption |
| CO2 | Per capita CO₂ emissions |

**Methodology:**

**Dimensionality Reduction:**
- Applied PCA (Principal Component Analysis)
- Standardized scaling for data normalization
- Identified optimal number of principal components

**Clustering Algorithms Compared:**
- K-Means
- Agglomerative Clustering (Hierarchical)
- DBSCAN
- Affinity Propagation
- Gaussian Mixture Models
- BIRCH

**Key Insights:**
- Countries naturally group by development level and resource management
- Economic indicators (PNB, Energy, CO₂) show strong correlation
- Environmental metrics (Deforestation, Forests) separate developed vs. developing nations
- **Central Question:** What can Mexico tell us about its cluster membership?

**Technologies:** Python, Scikit-learn, Scipy, Pandas, Matplotlib, Seaborn

---

## 🛠️ Technologies & Tools

- **Languages:** Python 3.10+, SQL
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, TensorFlow, PyTorch
- **Databases:** MySQL, PostgreSQL
- **Notebooks:** Jupyter Notebooks
- **Version Control:** Git & GitHub
- **Data Sources:** Kaggle, Maven Analytics, Custom datasets

---

## 📦 Project Structure

```
machine-learning-portfolio/
├── sql_superstore/
│   ├── README.md
│   ├── queries/
│   └── data/
├── coffee_sales/
│   ├── README.md
│   ├── coffee_sales.ipynb
│   └── data/
├── coffee_shop_sales/
│   ├── README.md
│   ├── coffee_shop_sales.ipynb
│   └── assets/
├── indicadores_economicos/
│   ├── README.md
│   ├── indicadores_economicos.ipynb
│   ├── Pais.csv
│   └── indicadores_economicos.html
├── car_price_prediction/
├── diabetes_detection/
├── fraud_detection/
├── supermarket_sales/
└── README.md
```

---

## 🎓 Learning & Development

This portfolio demonstrates proficiency in:

✅ End-to-end ML project lifecycle  
✅ Data preprocessing and feature engineering  
✅ Statistical analysis and hypothesis testing  
✅ Model selection and hyperparameter tuning  
✅ Time series analysis and forecasting  
✅ Unsupervised learning and pattern discovery  
✅ SQL database querying and optimization  
✅ Data visualization and storytelling  
✅ Technical documentation and communication  

---

## 📬 Contact & Connect

- 📧 **Email:** [luisgra@tuta.io]
- 💼 **LinkedIn:** [https://www.linkedin.com/in/lgra1525/](https://www.linkedin.com/in/lgra1525/)
- 🐱 **GitHub:** [https://github.com/sirlluis](https://github.com/sirlluis)
- 📊 **Kaggle:** [https://www.kaggle.com/luisgerardoram](https://www.kaggle.com/luisgerardoram)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated:** June 2026  
**Status:** 🚀 Actively maintained and expanding with new projects
