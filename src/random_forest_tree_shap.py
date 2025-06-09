import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import shap
import os

# =============================
# 1. Load Dataset
# =============================

# Load your pre-downloaded CSV dataset
df = pd.read_csv(r"../data/adult_income.csv")

# Example: if using Folktables ACSIncome dataset
# Binary target: income > 50K
df = df[df['PINCP'].notna()]
df['incg50'] = (df['PINCP'] > 50000).astype(int)
target_col = 'incg50'
X = df.drop(columns=[target_col, 'PINCP'])  # Drop label and original income
y = df[target_col]

# =============================
# 2. Preprocessing
# =============================

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include='number').columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
], remainder='passthrough')  # Keep numerical features as-is

# =============================
# 3. Train-Test Split
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =============================
# 4. Pipeline and Training
# =============================

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# =============================
# 5. Evaluation
# =============================

y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# =============================
# 6. SHAP Explanation
# =============================

# Extract trained model and preprocessed features
rf_model = pipeline.named_steps['classifier']
X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)

# Get feature names
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)

# SHAP TreeExplainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train_df)

# =============================
# 7. SHAP Visualization
# =============================

os.makedirs("shap_plots", exist_ok=True)

# Summary plot
shap.summary_plot(shap_values[1], X_train_df, show=False)
plt.title("TreeSHAP Summary Plot (Class 1 - income >50K)")
plt.savefig("shap_plots/summary_plot_class1.png")
plt.close()

# Bar plot of average absolute SHAP values
shap.summary_plot(shap_values[1], X_train_df, plot_type="bar", show=False)
plt.title("Feature Importance (SHAP)")
plt.savefig("shap_plots/feature_importance_class1.png")
plt.close()

print("SHAP plots saved to 'shap_plots/' directory.")

