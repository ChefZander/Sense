import tensorflow as tf
import numpy as np
import sensenet

def train_network():
    """
    Creates the TensorFlow network, trains it (with dummy data),
    and exports the weights.
    """
    # Define the network architecture
    model = sensenet.get_network()

    # --- Dummy Data for Demonstration ---
    # In a real scenario, you would load your actual training data here.
    x_train = np.random.rand(100, 768).astype(np.float32)
    y_train = np.random.randint(0, 2, (100, 1)).astype(np.float32)

    print("\nTraining the model with dummy data...")
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
    print("Training complete.")

    return model

if __name__ == "__main__":
    trained_model = train_network()
    trained_model.save("nn-1.keras")