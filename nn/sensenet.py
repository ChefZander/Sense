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

def board_to_bitboards(board: chess.Board) -> list[int]:
        bb = [0] * 768 # Initialize all to 0

        # Mapping piece types to indices (0-5 for P, N, B, R, Q, K)
        # This aligns with chess::PieceType(chess::PieceType::underlying(piecePlane % 6))
        piece_type_to_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }

        # The C++ code's `piecePlane` interpretation:
        # `piecePlane % 6` gives the piece type index (0-5)
        # `piecePlane > 6` for color indicates "current player's color" for `piecePlane` 7-11
        # `piecePlane <= 6` for color indicates "opponent's color" for `piecePlane` 0-6

        # This is a rather unusual way to define piece planes.
        # Let's clarify the 12 planes:
        # Planes 0-5: Opponent's pieces (P, N, B, R, Q, K)
        # Planes 6-11: Current player's pieces (P, N, B, R, Q, K)

        # Let's adjust for this specific C++ logic:
        # If `board.sideToMove()` is WHITE:
        #   piecePlane 0-5 are BLACK pieces (opponent)
        #   piecePlane 6-11 are WHITE pieces (current player)
        # If `board.sideToMove()` is BLACK:
        #   piecePlane 0-5 are WHITE pieces (opponent)
        #   piecePlane 6-11 are BLACK pieces (current player)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_type_idx = piece_type_to_idx[piece.piece_type]
                
                # Determine the piecePlane based on color and side_to_move
                if piece.color == board.turn: # Current player's piece
                    piece_plane = piece_type_idx
                else: # Opponent's piece
                    piece_plane = 6 + piece_type_idx
                
                # Set the bitboard value for the corresponding square
                bb[piece_plane * 64 + square] = 1
        
        return bb