import numpy as np
import matplotlib.pyplot as plt
#---------------------------game-------------------------------
def get_result(my_state, opponent_state, balance, prnt=False):
    loss, win = -1, 1
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
    return balance

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

#---------------------------train/test-------------------------------
def train_model(model, n, start_state, opponent_probabilities, states, prnt=False):
    prev_move = start_state
    for new_move in yield_opponent_moves(n, opponent_probabilities, prev_move, states, prnt):
        model.train_model(prev_move, new_move)
        if prnt:
            print(model)
            print()
        prev_move=new_move

def test_model(model, n, start_state, opponent_probabilities, states, prnt=False):
    results = []
    prev_move = start_state
    balance = 0
    results.append(balance)
    for opp_move in yield_opponent_moves(n, opponent_probabilities, prev_move, states, prnt):
        model_move = model.play(prev_move)
        balance = get_result(model_move, opp_move, balance, prnt)
        results.append(balance)

    return results
#------------------------------plot----------------------------------
def plot_results(results):
    plt.plot(results)
    plt.title("Balance vs games played")
    plt.xlabel("Games played")
    plt.ylabel("Balance")
    r = max(results) - min(results)
    plt.yticks(range(round(min(results)-int(r/10)), max(results)+int(r/10), 5))
    plt.axhline(results[-1])
    plt.text(0, results[-1]+1, f"final: {results[-1]}")
    plt.axhline(0)
    plt.savefig("plot.png")
    plt.show()

def summarize_results(results):
    losses= [results[x] for x in range(len(results)-1)  if results[x+1]<results[x]]
    wins= [results[x] for x in range(len(results)-1) if results[x+1]>results[x]]
    draws = [results[x] for x in range(len(results)-1) if results[x+1]==results[x]]
    print("Losses: {}".format(len(losses)))
    print("Wins: {}".format(len(wins)))
    print("Draws: {}".format(len(draws)))
    print("Total: {}".format(len(losses)+len(wins)+len(draws)))

def get_avg_results(n_tests, test_size, model, start_state, opponent_probabilities, states, prnt=False):
    results = []
    for i in range(n_tests):
        test = test_model(model, test_size, start_state, opponent_probabilities, states, prnt)
        results.append(test[-1])
    avg = sum(results)/len(results)
    print(f"Average results after {n_tests} tests: {avg}")
    return