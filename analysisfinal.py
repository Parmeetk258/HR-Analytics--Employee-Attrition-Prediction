#IMPORT THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#LOAD THE DATASET
df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

#CHECK THAT THE DATA LOADS CORRECTLY
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.shape)
print(df.describe())
print(df.columns)

#EXPLORATORY DATA ANALYSIS
#1.ATTRITION COUNT
plt.figure(figsize=(6,4))
sns.countplot(x="Attrition",data=df)
plt.title("Employee Attrition Count")
plt.show()
df.info()

#2.ATTRITION BY DEPARTMENT 
plt.figure(figsize=(8,5))
sns.countplot(x="Department",hue="Attrition",data=df)
plt.title("Department-wise Attrition")
plt.xticks(rotation=15)
plt.show()
df.info()

#3.ATTRITION BY GENDER
plt.figure(figsize=(6,4))
sns.countplot(x="Gender",hue="Attrition",data=df)
plt.title("Gender-wise Attrition")
plt.show()
df.info()

#4.ATTRITION BY OVERTIME
plt.figure(figsize=(6,4))
sns.countplot(x="OverTime",hue="Attrition",data=df)
plt.title("OverTime vs Attrition")
plt.show()
df.info()

#5.Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20,kde=True)
plt.title("Age Distribution")
plt.show()
df.info()

#6.Monthly Income Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['MonthlyIncome'], bins=20,kde=True)
plt.title("Monthly Income Distribution")
plt.show()
df.info()

#7.Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(numeric_only=True),annot=False ,cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
df.info()

#IMPORTING MACHINE LEARNING LIBRARIES
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder

df.dtypes
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

#CHECK COLUMNS
print(df.columns)
print(df.columns.tolist())

#FEATURES AND TARGET
X = df.drop('Attrition', axis=1)

y = df['Attrition']

#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

#LOGISTIC REGRESSION MODEL
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#ACCURACY
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print(classification_report(y_test, y_pred))

# ================= SHAP ANALYSIS =================

import shap
import matplotlib.pyplot as plt

# SHAP Explainer for Logistic Regression
explainer = shap.LinearExplainer(model, X_train)

# SHAP Values
shap_values = explainer.shap_values(X_test)

# ---------------- Feature Importance ----------------
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()

# ---------------- Summary Plot ----------------
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.show()

# ---------------- Waterfall Plot ----------------
exp = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns
)

shap.plots.waterfall(exp)
plt.show()