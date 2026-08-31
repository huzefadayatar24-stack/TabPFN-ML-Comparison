# TabPFN-ML-Comparison
# TabPFN vs Classical Models: Tabular Data Pipeline

An end-to-end machine learning pipeline comparing traditional tree-based and linear models against **TabPFN**, a foundational transformer model designed specifically for tabular data. 

This project features a fully interactive web interface and integrates **SHAP (SHapley Additive exPlanations)** to provide transparent, visual feature importance for the model's predictions.

## 🚀 Features
- **Automated Preprocessing:** Handles missing values and categorical encoding dynamically.
- **Model Benchmarking:** Compares Logistic Regression, Random Forest, and XGBoost against TabPFN.
- **Explainable AI (XAI):** Generates SHAP summary plots to visually interpret model behavior.
- **Interactive UI:** Built with Gradio for seamless CSV uploads and real-time inference.

## 🛠️ Tech Stack
- **Machine Learning:** `scikit-learn`, `xgboost`, `tabpfn`
- **Explainability:** `shap`
- **Data Manipulation:** `pandas`
- **Frontend/Deployment:** `gradio`

## ⚙️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/TabPFN-ML-Comparison.git](https://github.com/YOUR-USERNAME/TabPFN-ML-Comparison.git)
