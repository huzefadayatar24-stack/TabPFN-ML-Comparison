import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tabpfn import TabPFNClassifier
import shap
import matplotlib.pyplot as plt
import gradio as gr
import tempfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tabpfn import TabPFNClassifier
import shap
import matplotlib.pyplot as plt
import gradio as gr
import tempfile
import os

def run_project(train_file, target_column):
    df = pd.read_csv(train_file.name)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X = X.fillna(0)
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    xg = xgb.XGBClassifier(random_state=42)
    xg.fit(X_train, y_train)
    
    tab_model = TabPFNClassifier(device='cpu', N_ensemble_configurations=1)
    tab_model.fit(X_train, y_train)
    
    results_text = f"""
    Accuracy Comparison:
    - Logistic Regression: {lr.score(X_test, y_test)*100:.1f}%
    - Random Forest: {rf.score(X_test, y_test)*100:.1f}%
    - XGBoost: {xg.score(X_test, y_test)*100:.1f}%
    - TabPFN (Transformer): {tab_model.score(X_test, y_test)*100:.1f}%
    """
    
    explainer = shap.TreeExplainer(xg)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    
    temp_dir = tempfile.mkdtemp()
    graph_path = os.path.join(temp_dir, "shap_graph.png")
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()
    
    return results_text, graph_path

interface = gr.Interface(
    fn=run_project,
    inputs=[
        gr.File(label="Upload CSV Dataset"),
        gr.Textbox(label="Target Column Name")
    ],
    outputs=[
        gr.Textbox(label="Accuracy Results"),
        gr.Image(label="SHAP Feature Importance")
    ],
    title="TabPFN vs Classical Models"
)

if __name__ == "__main__":
    interface.launch()
