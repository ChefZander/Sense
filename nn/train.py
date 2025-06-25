import tensorflow as tf
import numpy as np
import sensenet

import chess # Import the python-chess library
import chess.engine
import pandas as pd
import random # For making random moves
import os # To check for file existence
from tensorflow.keras.layers import Input, Dense
import h5py
from tensorflow.keras.models import Model
import base64
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def simulate_single_game(engine_path, depth):
    """
    Simulates a single chess game and returns the collected states and targets.
    This function will be run concurrently.
    """
    board = chess.Board()
    game_states_and_evals = [] # To store (board_vector, normalized_score) for each move

    # Re-initialize engine for each process if needed, or pass it carefully
    # It's generally safer to open/close engines per process or use a manager.
    # For simplicity, we'll open/close for each game here, but for many games,
    # it's better to manage a pool of engines.
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        limit = chess.engine.Limit(depth=depth)

        # Initial random moves to diversify the game openings
        for _ in range(random.randint(5, 8)): # Random opening length
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            random_move = random.choice(legal_moves)
            board.push(random_move)
            # No evaluation here, just diversifying the start

        # Continue with engine moves until game over
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            try:
                # Get engine's best move
                result = engine.analyse(board, limit=limit)
                #print(result)
                engine_move = result.get('pv')[0]
            except chess.engine.EngineError:
                # Handle cases where the engine might not respond or fails
                print(f"Engine error, breaking game.")
                break

            # Evaluate the board BEFORE making the move
            # The score is from the perspective of the player whose turn it is.
            # .pov(True) gives score from White's perspective.
            # .winning_chance() converts centipawns/mate score to a 0-1 probability.
            normalized_score = result.get("score").wdl().pov(True).winning_chance()
            #print(normalized_score)
            game_states_and_evals.append((np.array(sensenet.board_to_bitboards(board.copy())), normalized_score))

            board.push(engine_move)

        # Game is over, append final state (if applicable and desired for evaluation)
        if not board.is_game_over(): # In case of early break
            info = engine.analyse(board, limit=limit)
            normalized_score = info.get("score").wdl().pov(True).winning_chance()
            game_states_and_evals.append((np.array(sensenet.board_to_bitboards(board.copy())), normalized_score))


        result = board.result()
        # Return results to be aggregated in the main process
        return game_states_and_evals, result

def decompress_board_vector(compressed_string: str) -> np.ndarray:
    """
    Converts a compressed string of '0's and '1's back into a NumPy array of integers.

    Args:
        compressed_string (str): The string representation of the board vector (e.g., "00101010").

    Returns:
        np.ndarray: A NumPy array containing 0s and 1s.
    """
    return np.array([int(char) for char in compressed_string])

# --- Main data generation function ---
def generate_chess_data_concurrent(engine_path, num_games, depth=10, max_workers=None, eval_scale = 400):
    """
    Generates training data by simulating random chess games concurrently.

    Args:
        engine_path (str): Path to the Stockfish engine executable.
        num_games (int): The number of chess games to simulate.
        depth (int): The search depth for the Stockfish engine.
        max_workers (int, optional): The maximum number of processes to use.
                                     Defaults to os.cpu_count() if None.

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - all_inputs (np.ndarray): 2D array of board state vectors.
               - all_targets (np.ndarray): 1D array of corresponding target outcomes (0 to 1).
    """
    all_inputs = []
    all_targets = []
    w, l, d = 0, 0, 0
    total_moves = 0
    start_time = time.time()

    print(f"\nGenerating training data from {num_games} chess games concurrently...")

    # Using ProcessPoolExecutor for true parallelism
    # max_workers=None will default to os.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(simulate_single_game, engine_path, depth): i for i in range(num_games)}

        game_count = 0
        for future in as_completed(futures):
            game_count += 1
            try:
                game_states_and_evals, result = future.result()

                if result == "1-0":
                    result_score = 1
                    w += 1
                elif result == "0-1":
                    result_score = 0
                    l += 1
                else:
                    result_score = 0.5
                    d += 1

                # Iterate through your game states and evaluations
                for board_vector, normalized_score in game_states_and_evals:
                    score = round(0.90 * normalized_score + 0.10 * result_score, 4)

                    # Accumulate in temporary lists (optional, but good if you also need them in memory)
                    all_inputs.append(board_vector)
                    all_targets.append(score / eval_scale)
                total_moves += len(game_states_and_evals)

                elapsed = time.time() - start_time
                print(f"  Game {game_count}/{num_games} simulated. Total states collected: {len(all_inputs)}. WLD: {w}/{l}/{d}. Moves/s: {total_moves/elapsed:.2f}")

            except Exception as exc:
                print(f"  Game generation generated an exception: {exc}")

    print(f"\nFinished generating data. Total states: {len(all_inputs)}")
    return np.array(all_inputs, dtype=np.float32), np.array(all_targets, dtype=np.float32)

def train_network():
    """
    Creates the TensorFlow network, trains it (with dummy data),
    and exports the weights.
    """
    # Define the network architecture
    model = sensenet.get_network()
    while True:
        X_train_resume, y_train_resume = generate_chess_data_concurrent(engine_path="/usr/bin/stockfish", depth=10, max_workers=4, num_games=10, eval_scale=400)
        epochs_resume = 5
        batch_size_resume = 16

        # Resume training and save the updated weights
        history = model.fit(
            X_train_resume, y_train_resume, 
            epochs=epochs_resume, 
            batch_size=batch_size_resume, 
        )
        loss = history.history.get('loss') 
        open("traininglosses.csv", "a").write(f"{loss}\n")
        model.save("nn.keras")


if __name__ == "__main__":
    train_network()