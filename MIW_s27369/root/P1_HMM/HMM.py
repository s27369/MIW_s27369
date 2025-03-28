import numpy as np
class HMM:
    def __init__(self):
        self.states = ["Rock", "Paper", "Scissors"]
        self.probabilities = {}
        self.observations = {}
        for state in self.states:
            self.probabilities[state] = {x:1/len(self.states) for x in self.states} #makes moves based on probabilities
            #opponent actions
            self.observations[state] = {x:0 for x in self.states}

    #---------------------------train---------------------------------
    def train_model(self, prev_move, new_move):
        self.record_observation(prev_move, new_move)
        self.update_probabilities(prev_move, new_move)
    def record_observation(self, prev_move, new_move):
        self.observations[prev_move][new_move] += 1

    def update_probabilities(self, prev_move, new_move):
        total = sum(self.observations[prev_move].values())
        if total > 0:
            self.probabilities[prev_move] = {
                x: self.observations[prev_move][new_move] / total
                for x in self.probabilities[prev_move].keys()
            }
        else:
            raise ZeroDivisionError("Total of row {} = 0".format(prev_move))
    #---------------------------move---------------------------------
    def play(self, prev_opp_move):
        predicted_opp_move = self.predict_opponent_move(prev_opp_move)
        return self.get_counter(predicted_opp_move)
    def predict_opponent_move(self, prev_move):
        return np.random.choice(self.states, p=list(self.probabilities[prev_move].values()))

    def get_counter(self, state):
        if state == "Rock":
            return "Paper"
        elif state == "Paper":
            return "Scissors"
        elif state == "Scissors":
            return "Rock"
        else:
            raise AttributeError("invalid state: {}".format(state))

    def __str__(self):
        msg = "HMM {\nprobabilities:\n"
        for state in self.probabilities.items():
            msg+=f"{state}\n"
        return msg+"}"


