import numpy as np
import random

balance = 0
loss = -1
win = 1

state = "Rock"

states = ["Rock", "Paper", "Scissors"]
rock_num = states.index("Rock")
paper_num = states.index("Paper")
scissors_num = states.index("Scissors")

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


# probabilities = {
#     "Rock": np.array([0.2, 0.6, 0.2]),
#     "Paper": np.array([0.2, 0.5, 0.3]),
#     "Scissors": np.array([0.2, 0.1, 0.7])
# }
#rozkład prawdopodobieństwa przeciwnika (model o tym nie wie)
probabilities = {
    "Rock": {"Rock":0.2, "Paper":0.6, "Scissors":0.2},
    "Paper": {"Rock":0.2, "Paper":0.5, "Scissors":0.3},
    "Scissors": {"Rock":0.2, "Paper":0.1, "Scissors":0.7}
}

def occurences_to_numpy(occurences:dict):
    return np.array(
        [[v for v in x.values()] for x in occurences.values()]
    )



def generate_previous_games(n:int) -> dict:
    global state
    occurences = {x: {z: 0 for z in y} for x, y in probabilities.items()}
    # occurences = {x: np.zeros(3) for x, y in probabilities.items()}
    prev = state
    for i in range(n):
        curr = get_opponent_move(prev)
        occurences[prev][curr]+=1
        prev=curr
    return occurences

def get_opponent_move(prev_move:str)->str:
    return np.random.choice(states, p=list(probabilities[prev_move].values()))



if __name__ == "__main__":
    prev_games = generate_previous_games(1000)
    print("Previous games:")
    for k in prev_games.items():
        print(k)
    print()


    print([(x.values()) for x in prev_games.values()])

