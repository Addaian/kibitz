import json
from parser import *
from pathlib import Path
import chess
import chess.engine

#Here, we load the full pipeline. This is where our entire system runs altogether.
'''
This is where we are putting our full kibitz pipeline.
Pipeline looks like this: 
PGN File Load -> PGN Parser -> PGN Extractor -> Prompt Builder -> Fine-tuned LLM -> Final Commentary Output
'''

stockfish = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")

def compute_eval(move : dict) -> None:
    # refine move takes a move and spits out the refined move, without changing the location of any values. 
    # uses Stockfish at a good depth of 18
    board_before = chess.Board(move["fen_before"]) # get the board state before as an object
    info_before = stockfish.analyse(board_before, chess.engine.Limit(depth=18))
    eval_before = info_before["score"].white.score(mate_score=10000)

    board_after = chess.Board(move["fen_after"])
    info_after = stockfish.analyse(board_after, chess.engine.Limit(depth=18))
    eval_after = info_after["score"].white.score(mate_score=10000)

    move["eval"] = eval_after - eval_before

def detect_phase(move : dict) -> None:     
    # really basic phase detection; could enhance in the future to detect whether or not out of book move? but line between middle and endgame is pretty clear.                                                                                             
    board = chess.Board(move["fen_before"])                                                                                              
    if board.fullmove_number <= 10:                                                                                       
        move["game_phase"] = "opening"                                                                                                 
    total = 0                                                                                                                 
    for color in chess.COLORS:                                                                                                
        total += len(board.pieces(chess.QUEEN, color)) * 9                                                                    
        total += len(board.pieces(chess.ROOK, color)) * 5                                                                     
        total += len(board.pieces(chess.BISHOP, color)) * 3                                                                   
        total += len(board.pieces(chess.KNIGHT, color)) * 3                                                                                    
    if total > 20:
        move["game_phase"] = "middlegame"                                                                                               
    else:
        move["game_phase"] = "endgame"

def classify_move(move : dict) -> None:
    # another really easy classification method
    if move["eval"] < -150:
        move["move_classification"] = "blunder"                                                                                               
    elif move["eval"] < -50:
        move["move_classification"] = "mistake"                                                                                                  
    elif move["eval"] < -30:
        move["move_classification"] = "inaccuracy"                                                                                               
    elif move["eval"] > 100:
        move["move_classification"] = "brilliant"                                                                                                
    elif move["eval"] >= 0:
        move["move_classification"] = "good"
    else:
        move["move_classification"] = "good"

if __name__ == "__main__":
    # step 1, load the pgn files. Here, we took each directory in /raw/ and appended them to a list in a Path structure. 
    pgn_files = []
    for folder in Path("data/raw").iterdir():
        if folder.is_dir():
            pgn_files.append(folder)

    # step 2, parse the pgns and store them in processed
    for folder in pgn_files:
        for pgn_file in folder.glob('*.pgn'):
            parse_pgn(pgn_file)

    # step 3, refine the pgn. By refine, we mean to enhance the features of each data. 
    for jsonl_file in Path("data/processed").glob("*.jsonl"):                                                           
        with open(jsonl_file) as f:
            for line in f:                                                                                              
                game = json.loads(line)
                for move in game["moves"]:
                    # move has: san, uci, fen_before, fen_after, comment, nags, eval 
                    compute_eval(move) # adds the eval value to each move (how good the move itself is)
                    detect_phase(move) # adds the phase that each move in each game is in
                    classify_move(move) # classifies into blunder, good, brilliant etc using eval of move                   
                    pass
                

 


    