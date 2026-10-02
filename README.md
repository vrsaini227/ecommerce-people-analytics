# 📊 E-Commerce People Analytics

## MCA Major Project

E-Commerce People Analytics is a data analytics and business intelligence
system designed to analyse customer behaviour, product purchasing patterns,
customer segments and unusual behavioural patterns using e-commerce data.

The system collects data through APIs, performs data cleaning and integration,
conducts exploratory and statistical analysis, applies machine-learning based
customer segmentation and behavioural anomaly detection, and presents the
results through an interactive Streamlit dashboard.

---

## 🎯 Project Objectives

The main objectives of the project are:

1. Collect e-commerce data using APIs.
2. Clean and preprocess raw datasets.
3. Integrate customer, product and cart information.
4. Perform exploratory data analysis.
5. Perform statistical validation of observed patterns.
6. Segment customers using machine learning.
7. Identify important product and category patterns.
8. Detect unusual customer behaviour.
9. Generate business-oriented analytical insights.
10. Present analytical results through an interactive dashboard.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- SciPy
- Streamlit
- REST API
- Jupyter/VS Code

---

## 📂 Project Structure

```text
E-Commerce People Analytics/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── api_fetch.py
│   ├── data_inspection.py
│   ├── clean_data.py
│   ├── integrate_data.py
│   ├── eda.py
│   ├── advanced_eda.py
│   ├── statistical_analysis.py
│   ├── customer_segmentation.py
│   ├── pattern_analysis.py
│   ├── anomaly_detection.py
│   └── business_insights.py
│
└── reports/
    ├── figures/
    └── business_insights.txt

*How to run Project*
1. Create Virtual Environment: 
    python -m venv venv
2. Activate Environment
    .\venv\Scripts\Activate.ps1
3. Install dependencies
    pip install -r requirements.txt
4. Run dashboard
    streamlit run app.py 
            or
    & ".\venv\Scripts\python.exe" -m streamlit run app.py