from data_preprocessing import (
    load_data,
    preprocess_data
)

from model import build_model

from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = load_data("data/customer_churn.csv")

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Build model
model = build_model(X_train.shape[1])

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop]
)

# Save model
model.save("outputs/model.h5")

print("Training completed successfully!")

import matplotlib.pyplot as plt

# Plot Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(
    ['Train', 'Validation']
)

plt.savefig("outputs/accuracy_plot.png")

plt.close()

# Plot Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(
    ['Train', 'Validation']
)

plt.savefig("outputs/loss_plot.png")

plt.close()

print("Graphs saved successfully!")