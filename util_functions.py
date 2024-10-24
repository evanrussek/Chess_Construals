import pandas as pd
import numpy as np

# remove first move of each player (this is the first two move_plys - half-moves - of the game)
def filter_moves(game_data):
    filt_game_data = game_data.loc[game_data.move_ply > 15].reset_index(drop = True)
    return filt_game_data

# compute rt for each move, add this to the dataframe (new name - data_rt)
def add_rt(game_df):
    
    game_df = game_df.reset_index(drop = True)
    game_time_setting = game_df.time_control[0] # this is the time_control (no clue what the 0 index is doing here)
    time_back = 0

    if '+1' in game_time_setting:
        time_back = 1
    elif '+2' in game_time_setting:
        time_back = 2
    elif '+3' in game_time_setting:
        time_back = 3
    elif '+5' in game_time_setting:
        time_back = 5
    elif '+10' in game_time_setting:
        time_back = 10
    elif '+20' in game_time_setting:
        time_back = 20
    
    white_move_rts = time_back + game_df.loc[game_df.white_active == True, 'clock'].diff()*-1
    black_move_rts = time_back + game_df.loc[game_df.white_active == False, 'clock'].diff()*-1
    game_df.loc[game_df.white_active == True, 'rt'] = white_move_rts
    game_df.loc[game_df.white_active == False, 'rt'] = black_move_rts
    
    return game_df