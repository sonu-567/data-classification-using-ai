
# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris

# Step 2: Load Dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Dataset Preview:\n", df.head())

# Step 3: Visualization (Feature Distribution)
df.hist(figsize=(10, 6))
plt.suptitle("Feature Distribution")
plt.show()

# Step 4: Split Data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Step 8: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Step 9: Feature Importance (Coefficients)
plt.bar(X.columns, model.coef_[0])
plt.title("Feature Importance (Class 0)")
plt.xticks(rotation=45)
plt.show()