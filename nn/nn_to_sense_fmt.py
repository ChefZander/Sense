import numpy as np
import tensorflow as tf
import sys

def export_weights_to_txt(model, filename="nn-1.sense"):
    """
    Exports the weights and biases of the TensorFlow model to a text file.
    The format will be:
    [layer_name]_weights:
    [weight_matrix_flat_values]
    [layer_name]_bias:
    [bias_vector_flat_values]
    """
    with open(filename, 'w') as f:
        for layer in model.layers:
            weights = layer.get_weights()
            if weights: # Check if the layer has weights (Dense layers will)
                # Weights (kernel)
                if len(weights) > 0:
                    f.write(f"{layer.name}_weights:\n")
                    # Flatten the weights and write them
                    np.savetxt(f, weights[0].flatten(), fmt='%f')

                # Biases
                if len(weights) > 1:
                    f.write(f"{layer.name}_bias:\n")
                    # Flatten the biases and write them
                    np.savetxt(f, weights[1].flatten(), fmt='%f')
    print(f"\nWeights exported to {filename}")

model = tf.keras.models.load_model(sys.argv[1])
model.summary()
export_weights_to_txt(model, filename=sys.argv[2])