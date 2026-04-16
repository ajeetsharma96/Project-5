import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

from data_preprocessing import (
    load_data,
    preprocess_data
)

# Load dataset
df = load_data("data/customer_churn.csv")

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Load trained model
model = load_model("outputs/model.h5")

# Predict
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")

plt.close()

# Print classification report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred_classes
    )
)

print("\nEvaluation completed!")