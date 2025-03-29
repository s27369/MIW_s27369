from functions import *
from HMM import *
balance = 0
loss = -1
win = 1

states = ["Rock", "Paper", "Scissors"]

# rock_num = states.index("Rock")
# paper_num = states.index("Paper")
# scissors_num = states.index("Scissors")

# probabilities = {
#     "Rock": np.array([0.2, 0.6, 0.2]),
#     "Paper": np.array([0.2, 0.5, 0.3]),
#     "Scissors": np.array([0.2, 0.1, 0.7])
# }
#rozkład prawdopodobieństwa przeciwnika (model o tym nie wie)
opponent_probabilities = {
    "Rock": {"Rock":0.2, "Paper":0.6, "Scissors":0.2},
    "Paper": {"Rock":0.2, "Paper":0.5, "Scissors":0.3},
    "Scissors": {"Rock":0.2, "Paper":0.1, "Scissors":0.7}
}

if __name__ == "__main__":
    model = HMM()
    print(model)
    start_state = "Rock"
    train_model(model, 100000, start_state, opponent_probabilities, states, False)
    print(model)
    print()

    results = test_model(model, 1000, start_state, opponent_probabilities, states, False)
    plot_results(results)
