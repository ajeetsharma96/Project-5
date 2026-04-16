from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def build_model(input_dim):

    # Create model
    model = Sequential()

    # Input layer
    model.add(Dense(
        units=64,
        activation="relu",
        input_dim=input_dim
    ))

    # Dropout layer
    model.add(Dropout(0.3))

    # Hidden layer
    model.add(Dense(
        units=32,
        activation="relu"
    ))

    # Dropout layer
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(
        units=1,
        activation="sigmoid"
    ))

    # Compile model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model