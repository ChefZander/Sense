import chess
import math
import os

# Parameters
INPUT_NEURONS = 768
HL1_NEURONS = 16
OUTPUT_NEURONS = 1
QUANTIZATION = 255
EVAL_SCALE = 400

# Model architecture is
# IL (768) -> HL1 (16, no activation) -> OL (1, sigmoid)

hl1_weights = [0] * (INPUT_NEURONS * HL1_NEURONS)
hl1_bias = [0] * HL1_NEURONS

output_weights = [0] * (HL1_NEURONS * OUTPUT_NEURONS)
output_bias = [0] * OUTPUT_NEURONS

class SenseNet:
    @staticmethod
    def parse_line(line: str) -> list[int]:
        values = []
        for token in line.split():
            try:
                values.append(int(token))
            except (ValueError, IndexError) as e:
                print(f"Error parsing token '{token}': {e}", file=os.sys.stderr)
        return values

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def load_weights(file_path: str = "nn.sense"):
        global hl1_weights, hl1_bias, output_weights, output_bias

        if not os.path.exists(file_path):
            print(f"info string Error: Could not open {file_path}", file=os.sys.stderr)
            return

        current_section = ""
        weight_idx = 0

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                if line.endswith(':'):
                    current_section = line[:-1]
                    weight_idx = 0
                    continue

                values = SenseNet.parse_line(line)

                if current_section == "hidden_layer_1_weights":
                    for val in values:
                        if weight_idx < len(hl1_weights):
                            hl1_weights[weight_idx] = val
                            weight_idx += 1
                        else:
                            print(f"Warning: Exceeded hl1_weights capacity during loading.", file=os.sys.stderr)
                            break
                elif current_section == "hidden_layer_1_bias":
                    for val in values:
                        if weight_idx < len(hl1_bias):
                            hl1_bias[weight_idx] = val
                            weight_idx += 1
                        else:
                            print(f"Warning: Exceeded hl1_bias capacity during loading.", file=os.sys.stderr)
                            break
                elif current_section == "output_layer_weights":
                    for val in values:
                        if weight_idx < len(output_weights):
                            output_weights[weight_idx] = val
                            weight_idx += 1
                        else:
                            print(f"Warning: Exceeded output_weights capacity during loading.", file=os.sys.stderr)
                            break
                elif current_section == "output_layer_bias":
                    for val in values:
                        if weight_idx < len(output_bias):
                            output_bias[weight_idx] = val
                            weight_idx += 1
                        else:
                            print(f"Warning: Exceeded output_bias capacity during loading.", file=os.sys.stderr)
                            break
        print("info string Weights loaded successfully!")

    @staticmethod
    def board_to_bitboards(board: chess.Board) -> list[int]:
        bb = [0] * INPUT_NEURONS # Initialize all to 0

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

    @staticmethod
    def predict(input_data: list[int]) -> float:
        # HL1
        hl1_output = [0] * HL1_NEURONS
        for j in range(HL1_NEURONS):
            current_sum = 0
            for i in range(INPUT_NEURONS):
                if input_data[i] == 1:
                    current_sum += hl1_weights[i * HL1_NEURONS + j]
            hl1_output[j] = current_sum + hl1_bias[j]

        # OL
        output_sum = 0
        for i in range(HL1_NEURONS):
            output_sum += hl1_output[i] * output_weights[i]
        output_sum += output_bias[0]

        return (float(output_sum) / QUANTIZATION / QUANTIZATION) * EVAL_SCALE

# Example usage:
if __name__ == "__main__":
    SenseNet.load_weights()
    board = chess.Board()
    input_features = SenseNet.board_to_bitboards(board)
    print("".join(map(str, input_features)))

    prediction = SenseNet.predict(input_features)
    print(f"Prediction for the current board: {prediction:.2f}")