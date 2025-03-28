import numpy as np
import random

#---------------------------game-------------------------------
def get_result(my_state, opponent_state, prnt=False):
    global balance, loss, win
    if my_state == opponent_state:
        result = "Draw"
    elif (
            ((my_state == "Rock") and (opponent_state == "Paper")) or
            ((my_state == "Paper") and (opponent_state == "Scissors")) or
            ((my_state == "Scissors") and (opponent_state == "Rock"))
    ):
        result = "Loss"
        balance = balance + loss
    else:
        result = "Win"
        balance = balance + win

    if prnt: print(f'{result}! Balance: {balance}')

def get_counter(state):
    if state == "Rock":
        return "Paper"
    elif state == "Paper":
        return "Scissors"
    elif state == "Scissors":
        return "Rock"
    else:
        raise AttributeError("invalid state: {}".format(state))


#---------------------------move-------------------------------
def get_opponent_move(prev_move:str, probabilities, states)->str:
    return np.random.choice(states, p=list(probabilities[prev_move].values()))
def generate_previous_games(n:int, probabilities, starting_state, states) -> dict:
    occurences = {x: {z: 0 for z in y} for x, y in probabilities.items()}
    # occurences = {x: np.zeros(3) for x, y in probabilities.items()}
    prev = starting_state
    for i in range(n):
        curr = get_opponent_move(prev, probabilities, states)
        occurences[prev][curr]+=1
        prev=curr
    return occurences

def yield_opponent_moves(n:int, probabilities, starting_state, states, prnt=False):
    prev = starting_state
    for i in range(n):
        curr = get_opponent_move(prev, probabilities, states)
        prev = curr
        if prnt: print("opp move: {}".format(curr))
        yield curr
#---------------------------other-------------------------------
def occurences_to_numpy(occurences:dict)->np.array:
    return np.array(
        [[v for v in x.values()] for x in occurences.values()]
    )
#
# def normalize_occurences(occurences:np.array)->np.array:
#     result = np.array([
#         [],[],[]
#     ])
#
#     for row in occurences:
#         total = np.sum(row)
#         result