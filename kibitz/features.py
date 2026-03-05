import chess.pgn
# here, we use the chess package. This allows us to parse a PGN finely. 

# Open singular PGN file
def parseGame(PGNpath : str): 
    pgn = open(PGNpath) # this gets us the game pgn itself
    
