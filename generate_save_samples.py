import os
import chess
import chess.engine
import pandas as pd
import numpy as np
import json
import os 
import time

from sample_functions import *
from util_functions import *

is_array_job = True

if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    job_idx = 0

# SET UP PARAMETERS FOR THIS JOBS
sample_params = {}
sample_params['n_samples'] = 50 # 50 samples per move?
sample_params['n_alternate_moves'] = 5
sample_params['eval_depth_consideration'] = 13
sample_params['eval_depth_evaluation'] = 13
sample_params['p_drop'] = .35
sample_params['p_move'] = 0

param_folder_name = '_'.join([f"{key}-{value}" for key, value in sample_params.items()])


# Set up folders
working_folder = '/home/erussek/projects/Chess_Construals';
os.chdir(working_folder)

# use stockfish 14
stockfish_file = "stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2"

engine_folder = '/home/erussek/projects/utils'

stockfish_path = os.path.join(engine_folder, stockfish_file)
chess_data_folder = "/scratch/gpfs/erussek/Chess_project/Lichess_2019_data"


# maybe add something about the sampling settings here... 
to_save_folder = "/scratch/gpfs/erussek/Chess_Construals/sample_res/August_2019"
os.makedirs(to_save_folder, exist_ok = True)

to_save_folder_params = os.path.join(to_save_folder, param_folder_name)
os.makedirs(to_save_folder_params, exist_ok = True)

# set info on number of moves
moves_per_run = 1000 # see how long this takes... 
n_total_runs = 100  

run_by_idxs = np.reshape(np.arange(moves_per_run*n_total_runs), (n_total_runs, moves_per_run))

run_idxs = run_by_idxs[job_idx,:]

# load in data for this job - this should include both a folder and a # of rows
chess_csv_file = 'lichess_db_standard_rated_2019-08.csv' # this is the august games
chess_data_fullfile = os.path.join(chess_data_folder,chess_csv_file)
data_first = pd.read_csv(chess_data_fullfile, nrows=2)
cols = data_first.columns
data_raw = pd.read_csv(chess_data_fullfile, skiprows = run_idxs[0]+1, nrows=moves_per_run, names = cols)

# add rt and filter to relevant games... 
data_rt = data_raw.groupby('game_id').apply(add_rt).reset_index(drop = True)
    #filt_game_data = filt_game_data.iloc[:-10]

# remove first and last 12 moves of each game... anything else?
data_filt = data_rt.groupby('game_id').apply(filter_moves).reset_index(drop=True)


# what game types do we care about? 
game_time_types = ['60+0', '120+1', '180+0',
                   '180+2', '300+0', '300+3',
                   '600+0', '600+5', '900+10',
                   '1800+0', '1800+20']

# filter based on if it's in these game settings... 
data_filt = data_filt.loc[data_filt.time_control.isin(game_time_types)].reset_index(drop = True)
data = data_filt

# set up engine... 
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

move_res = []
iteration_times = []  # List to store the time of each iteration

# how many moves are left?
n_moves_remaining = data.shape[0]

for move_idx in range(n_moves_remaining):
    start_time = time.time()  # Record the start time

    print(move_idx, end = ' ')
    this_row = data.loc[move_idx,:]
    this_board_state = this_row.board
    true_board = chess.Board(this_board_state)
    move_selected_uci = this_row.move

    this_move_sample_res = compute_move_sample_res(sample_params, true_board, move_selected_uci, engine)
        
    move_res.append(this_move_sample_res)
    
    end_time = time.time()  # Record the end time
    iteration_time = end_time - start_time  # Calculate the duration of the iteration
    iteration_times.append(iteration_time)  # Append the time to the list
    
# SAVE THE DATA!!!! 

print('SAVING RESULTS')
start_idx = run_idxs[0];
end_idx = run_idxs[-1];

file_name = 'job_'+str(job_idx)+'_start_'+str(start_idx)+'_end_'+str(end_idx)+'.json'
full_to_save_name = os.path.join(to_save_folder_params, file_name)

with open(full_to_save_name, "w") as f:
    json.dump(move_res, f, indent=4)

# close the chess engine.
engine.quit()
print('script is done!')
    
# print("--- %s seconds ---" % (time.time() - start_time))