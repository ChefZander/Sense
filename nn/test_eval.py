import numpy as np
import tensorflow as tf
import sys
import sensenet
import chess

model = tf.keras.models.load_model(sys.argv[1])
model.summary()
# Get the board state as a list of floats
board_input_list = sensenet.board_to_bitboards(chess.Board())

# Convert the list to a NumPy array
board_input_array = np.array(board_input_list, dtype=np.float32)

# Add an extra dimension for the batch size.
# If board_input_array has shape (768,), this will become (1, 768).
batched_input = np.expand_dims(board_input_array, axis=0)

# Now, predict with the batched input
prediction = model.predict(batched_input)
print(prediction)