import chess
import chess.engine
import random
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

games = 100000

with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
    TimeElapsedColumn(),
    TextColumn("{task.fields[status]}"), # Custom column for dynamic status updates
    refresh_per_second=1 # Update the display 10 times per second
) as progress:
    
    # Create a task for the overall game progress
    games_task = progress.add_task("[green]Processing Games...", total=games, status="Starting")

    with chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish") as engine:
        limit = chess.engine.Limit(nodes=10000)
        
        for i in range(games):
            current_game_status = f"Game {i+1}/{games}"
            progress.update(games_task, status=f"{current_game_status}")

            with open("output.csv", "a") as h:
                board = chess.Board()
                # Make the first 10 moves random
                try:
                    for _ in range(10):
                        board.push(random.choice(list(board.legal_moves)))
                except Exception:
                    continue
                
                moves_task = progress.add_task(f"[cyan]Moves for Game {i+1}...", total=None, status="Thinking", visible=False) # Start invisible
                progress.update(moves_task, visible=True) # Make it visible when the game starts

                move_count = 0
                while not board.is_game_over():
                    try:
                        result = engine.analyse(board, limit=limit)
                        engine_move = result.get('pv')[0]
                        score = result.get("score").wdl().pov(True).winning_chance()
                        
                        h.write(f"{board.fen()},{score}\n")
                        board.push(engine_move)
                        move_count += 1
                        
                        # Update the inner progress bar and status
                        progress.update(moves_task, advance=1, description=f"[cyan]Moves for Game {i+1} [red]({move_count})", status=f"Score: {score:.2f}")
                        
                    except chess.engine.EngineError:
                        progress.update(moves_task, status="Engine Error!", completed=move_count)
                        print(f"\nEngine error in Game {i+1}, breaking.")
                        break
                
                progress.remove_task(moves_task) # Remove the inner task when the game is done
                progress.update(games_task, advance=1, status=f"{current_game_status} - Done! Final Score: {score:.2f}") # Update outer task