#### FUNCTIONS TO SAMPLE CONSTRUAL
import chess
import numpy as np

from scoring_functions import *

def get_neighbouring_squares(square):
    
    
    neighbours = []
    if square>=8:
        neighbours += [square-8]
    if square<56:
        neighbours += [square+8]
    if square%8>0:
        neighbours += [square-1]
    if square%8<7:
        neighbours += [square+1]
    return neighbours


def sample_construal(board,pdrop=0.5,pmove=0.2, move_selected = ''):
    
    # return a list of probabilities of each random choice made to get how likely the board was
    # given the sampling process - note that this is not exactly right b/c you reject bad boards
    
    # this is so as not to select the moved from square... , but i think it's fine to sample that
    
    # also count the number of pieces dropped vs not dropped and the number of pieces moved vs not moved
    

    while True:
        
        n_pieces_dropped = 0
        n_pieces_moved = 0
        n_pieces_not_dropped = 0
        n_pieces_not_moved = 0

        prob_sample_actions = [] 
        
        new_board = board.copy()
        piece_map = new_board.piece_map()
        
        if move_selected == '':
            moved_from_square = ''
        else:
            moved_from_square = chess.Move.from_uci(move_selected).from_square

        new_piece_map = piece_map.copy()
        
        #step 1: drop pieces - don't drop the kings or the piece that was moved
        for square in piece_map:
            
            # check that it's not a king - and also don't drop the square that was moved from (or do?)
            if (piece_map[square] != chess.Piece.from_symbol('K')) & (piece_map[square] != chess.Piece.from_symbol('k')) & (square != moved_from_square):
            
                if (np.random.binomial(1,pdrop)>0): # delete the piece
                    del new_piece_map[square]
                    prob_sample_actions.append(pdrop)
                    n_pieces_dropped+=1
                else:
                    prob_sample_actions.append(1 - pdrop)
                    n_pieces_not_dropped+=1

        #step 2: move pieces
        occupied_squares = list(new_piece_map.keys())
        np.random.shuffle(occupied_squares)

        for square in occupied_squares:
            # don't move piece that was moved # can you move the king????
            new_squares = get_neighbouring_squares(square)
            new_squares_legal = [ns for ns in new_squares if ns not in piece_map]
            if len(new_squares_legal) > 0:
                if np.random.binomial(1,pmove)>0 & (square != moved_from_square):
                    prob_sample_actions.append(pmove)
                    # update piece map
                    new_square = np.random.choice(new_squares_legal)
                    prob_sample_actions.append(1/len(new_squares_legal))   
                    new_piece_map[new_square] = new_piece_map.pop(square)
                    
                    n_pieces_moved+=1

                else:
                    prob_sample_actions.append(1 - pmove)
                    n_pieces_not_moved+=1
                    
        
        cost_info = {"n_pieces_not_dropped": n_pieces_not_dropped, "n_pieces_not_moved": n_pieces_not_moved, "n_pieces_dropped": n_pieces_dropped, "n_pieces_moved": n_pieces_moved}

        new_board.set_piece_map(new_piece_map)
        
        if new_board.is_valid():
            
            return new_board, prob_sample_actions, cost_info
        

# FUNCTIONS TO 

def get_construal_selected_consideration_set(c_i, engine, n_moves = 10, eval_depth = 10):

    # returns the moves selected in a construal and their (processed values)
    engine.configure({"Clear Hash": None})
    info = engine.analyse(c_i, chess.engine.Limit(depth=eval_depth), multipv=n_moves)
    # print(info)

    ci_selected_moves = []
    ci_selected_move_vals = []
    ci_selected_move_unprocessed_vals = []

    if isinstance(info, list):
        for i in range(len(info)):
            this_move = info[i]['pv'][0]
            ci_selected_moves.append(this_move.uci())
            
            this_score = info[i]['score']
            processed_score = process_score(this_score, c_i.turn)
            ci_selected_move_vals.append(processed_score)
            ci_selected_move_unprocessed_vals.append(this_score)
            
    else:
        this_move = info['pv'][0]
        ci_selected_moves.append(this_move.uci())
        
        this_score = info[i]['score']
        processed_score = process_score(this_score, c_i.turn)
        ci_selected_move_vals.append(processed_score)
        ci_selected_move_vals.append(processed_score)
        ci_selected_move_unprocessed_vals.append(this_score)

    return np.array(ci_selected_moves), np.array(ci_selected_move_vals), ci_selected_move_unprocessed_vals

def get_softmax_action_probs(action_values, B):
    
    return np.exp(B*action_values) / np.sum(np.exp(B*action_values))

# ci_consideration_set
# get construal utility

def get_construal_utility_max(c_i, true_board, engine, eval_depth = 10):
    
    # should only be getting legal boards, but if not, check...

    engine.configure({"Clear Hash": None})

    # get top move in the construal
    info = engine.analyse(c_i, chess.engine.Limit(depth=eval_depth))
    m_c_i = info['pv'][0]

    # now evaluate m_c_i on the true board...
    # is this a legal move
    if m_c_i in true_board.legal_moves:
        
        engine.configure({"Clear Hash": None})
        info = engine.analyse(true_board, chess.engine.Limit(depth=eval_depth), root_moves=[m_c_i])
        m_c_i_score_unprocessed = info['score']
        m_c_i_score_processed = process_score(m_c_i_score_unprocessed, true_board.turn)
        U_c_i = m_c_i_score_processed

    else:
        
        U_c_i = 0

    return U_c_i

# def get_prob_move_given_construal


def evaluate_chosen_move_and_alternates_in_c_i(c_i, move_selected_uci, engine, n_alternate_moves = 5, eval_depth_consideration = 10, eval_depth_evaluation = 10):

    # think about how many alternate moves to include, and what to call that (could include move made)
    
    move_selected_struc = chess.Move.from_uci(move_selected_uci)
    
    if move_selected_struc not in c_i.legal_moves:
        return "error" # figure out how to throw an error - this should be checked for before calling this func.
    
    # generate consideration set
    ci_consideration_set, _, _  = get_construal_selected_consideration_set(c_i, engine, n_moves = n_alternate_moves, eval_depth = eval_depth_consideration)
    
    # remove the move selected to make the accounting simpler...
    alternate_ci_moves = ci_consideration_set[ci_consideration_set != move_selected_uci]
    
    # place the selected move to be first in the list...
    move_selected_w_alternates = np.insert(alternate_ci_moves, 0, move_selected_uci)
    
    # now re-evaluate every-move, with the root-move trick. Resultant 
    move_selected_w_alternates_scores_unprocessed = []
    move_selected_w_alternates_scores_processed = []
    
    for m in move_selected_w_alternates:
    
        engine.configure({"Clear Hash": None})
        info = engine.analyse(c_i, chess.engine.Limit(depth=eval_depth_evaluation), root_moves=[chess.Move.from_uci(m)])
        score_unprocessed = info['score']
        move_selected_w_alternates_scores_unprocessed.append(score_unprocessed)
        
        score_processed = process_score(score_unprocessed, c_i.turn)
        
        # could just compute this...
        move_selected_w_alternates_scores_processed.append(score_processed)
        
    return np.array(move_selected_w_alternates_scores_processed), np.array(move_selected_w_alternates)
    

# check if the move is legal...

def compute_prob_move_g_construal(move_selected_uci, c_i, engine, beta_move = 500, n_alternate_moves = 5, eval_depth_consideration = 10, eval_depth_evaluation = 10):

    # consider whether you want to constrain the total size of the consideration set or keep it constant
    if chess.Move.from_uci(move_selected_uci) in c_i.legal_moves:

        # get value of each move. potentially save these for purposes of figuring out parameters... could use them alter on for optimization of 
        # beta
        move_selected_w_alternates_scores_processed, _ = evaluate_chosen_move_and_alternates_in_c_i(c_i, move_selected_uci, engine, n_alternate_moves = n_alternate_moves, eval_depth_consideration = eval_depth_consideration, eval_depth_evaluation = eval_depth_evaluation)

        # can save the log of this?
        move_lik = np.exp(beta_move*move_selected_w_alternates_scores_processed[0])/np.sum(np.exp(beta_move*move_selected_w_alternates_scores_processed))

    else:

        move_lik = 0
    
    return move_lik 

# compute_prob_move_g_construal('h3f3', c_i)


def compute_move_sample_res(sample_params, true_board, move_selected_uci):
    
    # Get parameters to run the samples
    n_samples = sample_params['n_samples']
    n_alternate_moves = sample_params['n_alternate_moves']
    eval_depth_consideration = sample_params['eval_depth_consideration']
    eval_depth_evaluation = sample_params['eval_depth_evaluation']
    p_drop = sample_params['p_drop']
    p_move = sample_params['p_move']
    
    # Data strucs to save results
    move_scores_res = np.zeros((n_samples, n_alternate_moves))
    n_pieces_res = np.zeros((n_samples))
    q_i_res = np.zeros((n_samples))
    U_c_i_res = np.zeros((n_samples))
    illegal_in_c_i_res = np.zeros((n_samples), dtype = bool)
    n_available_moves_res = np.zeros((n_samples))
    
    for sample_idx in range(n_samples):

        c_i, q_i_list, cost_info = sample_construal(true_board,pdrop=0.35,pmove=0)
        q_i = np.prod(q_i_list) # probability under sampling distribution
        U_c_i = get_construal_utility_max(c_i, true_board, engine, eval_depth = 10)
        n_pieces = cost_info['n_pieces_not_dropped']

        if chess.Move.from_uci(move_selected_uci) in c_i.legal_moves:
            move_selected_w_alternates_scores_processed, _ = evaluate_chosen_move_and_alternates_in_c_i(c_i, move_selected_uci, engine, n_alternate_moves = n_alternate_moves, eval_depth_consideration = eval_depth_consideration, eval_depth_evaluation = eval_depth_evaluation)

            n_available_moves = min(len(move_selected_w_alternates_scores_processed), n_alternate_moves)


            move_scores_res[sample_idx, :] = move_selected_w_alternates_scores_processed[:n_available_moves]
            n_pieces_res[sample_idx] = n_pieces
            q_i_res[sample_idx] = q_i
            U_c_i_res[sample_idx] = U_c_i
            n_available_moves_res[sample_idx] = n_available_moves
        else:
            illegal_in_c_i_res[sample_idx] = True

    this_move_sample_res = {'move_scores': move_scores_res,
                      'n_pieces': n_pieces_res,
                      'q_i': q_i_res,
                      'U_c_i': U_c_i_res,
                      'illegal_move': illegal_in_c_i_res}
    
    return this_move_sample_res