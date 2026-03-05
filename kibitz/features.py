from parser import *
from pathlib import Path

#Here, we load the full pipeline. This is where our entire system runs altogether.
'''
This is where we are putting our full kibitz pipeline.
Pipeline looks like this: 
PGN File Load -> PGN Parser -> PGN Extractor -> Prompt Builder -> Fine-tuned LLM -> Final Commentary Output
'''

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



    