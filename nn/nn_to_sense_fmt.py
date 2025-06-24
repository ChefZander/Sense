import numpy as np
import tensorflow as tf
import sys

def export_weights_to_txt(model, filename="nn-1.sense"):
    """
    Exports the weights and biases of the TensorFlow model to a text file.
    Each weight and bias value is multiplied by a QUANTIZATION constant
    and then rounded to the nearest integer.

    The format will be:
    [layer_name]_weights:
    [quantized_weight_matrix_flat_values]
    [layer_name]_bias:
    [quantized_bias_vector_flat_values]
    """
    # Define the quantization constant
    QUANTIZATION = 255

    with open(filename, 'w') as f:
        # Iterate through each layer in the model
        for layer in model.layers:
            # Get the weights and biases for the current layer
            weights = layer.get_weights()

            # Check if the layer has any weights (e.g., Dense layers will, Activation layers won't)
            if weights:
                # Process weights (kernel) if they exist
                if len(weights) > 0:
                    f.write(f"{layer.name}_weights:\n")
                    # Multiply by QUANTIZATION, round to nearest integer, and convert to int
                    # Flatten the array for saving to file
                    quantized_weights = np.round(weights[0] * QUANTIZATION).astype(int)
                    np.savetxt(f, quantized_weights.flatten(), fmt='%d') # Use %d for integer output

                # Process biases if they exist
                if len(weights) > 1:
                    f.write(f"{layer.name}_bias:\n")
                    # Multiply by QUANTIZATION, round to nearest integer, and convert to int
                    # Flatten the array for saving to file
                    quantized_biases = np.round(weights[1] * QUANTIZATION).astype(int)
                    np.savetxt(f, quantized_biases.flatten(), fmt='%d') # Use %d for integer output

    print(f"\nWeights exported to {filename}")

model = tf.keras.models.load_model(sys.argv[1])
model.summary()
export_weights_to_txt(model, filename=sys.argv[2])