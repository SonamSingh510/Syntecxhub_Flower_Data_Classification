import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset & EDA
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("--- Iris Dataset Loaded ---")
print(df.head())

# Visualize feature pairs (Pairplot)
sns.pairplot(df, hue='species', palette='viridis')
plt.show()

# 2. Train/Test Split
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train & Compare Models
# Model A: Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Model B: Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

print(f"\nLogistic Regression Accuracy: {accuracy_score(y_test, lr_preds):.2f}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_preds):.2f}")

# 4. Confusion Matrix (for Logistic Regression)
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Iris Species')
plt.show()

# 5. Small CLI for New Prediction
def predict_flower():
    print("\n--- Predict New Flower Species ---")
    try:
        sl = float(input("Enter Sepal Length (cm): "))
        sw = float(input("Enter Sepal Width (cm): "))
        pl = float(input("Enter Petal Length (cm): "))
        pw = float(input("Enter Petal Width (cm): "))
        
        features = np.array([[sl, sw, pl, pw]])
        prediction = lr_model.predict(features)
        print(f"Result: The species is likely '{iris.target_names[prediction[0]]}'")
    except ValueError:
        print("Invalid input. Please enter numeric values.")

predict_flower()