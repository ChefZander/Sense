import tensorflow as tf
import chess

def get_network(): 
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(768,), use_bias=True, name='hidden_layer_1'),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True, name='output_layer')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def board_to_bitboards(board: chess.Board) -> list[float]:
    """
    Converts a chess.Board object into a list of 12 bitboards.

    The order of bitboards in the returned list is crucial:
    [0]  Side to move Pawns
    [1]  Side to move Knights
    [2]  Side to move Bishops
    [3]  Side to move Rooks
    [4]  Side to move Queens
    [5]  Side to move Kings
    [6]  Side NOT to move Pawns
    [7]  Side NOT to move Knights
    [8]  Side NOT to move Bishops
    [9]  Side NOT to move Rooks
    [10] Side NOT to move Queens
    [11] Side NOT to move Kings

    Each bitboard is a 64-bit integer where a set bit (1) at position 'n'
    indicates the presence of the corresponding piece type on the square
    represented by 'n' (a1=0, b1=1, ..., h8=63).

    Args:
        board (chess.Board): The chess.Board object to convert.

    Returns:
        list[int]: A list of 12 integers, each representing a bitboard.
                   The integers are treated as 64-bit unsigned values.
    """
    intermediate_bitboards = [0] * 12
    piece_type_to_index = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    side_to_move_color = board.turn
    side_not_to_move_color = not board.turn

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece:
            piece_type_idx = piece_type_to_index[piece.piece_type]
            bit_position = square

            if piece.color == side_to_move_color:
                intermediate_bitboards[piece_type_idx] |= (1 << bit_position)
            elif piece.color == side_not_to_move_color:
                intermediate_bitboards[piece_type_idx + 6] |= (1 << bit_position)

    flattened_input = []
    for bb in intermediate_bitboards:
        for i in range(64):
            bit_value = (bb >> i) & 1
            flattened_input.append(float(bit_value))

    return flattened_input