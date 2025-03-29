from functions import *
from HMM import *

states = ["Rock", "Paper", "Scissors"]

# rozkład prawdopodobieństwa przeciwnika (model o tym nie wie)
opponent_probabilities = {
    "Rock": {"Rock": 0.2, "Paper": 0.6, "Scissors": 0.2},
    "Paper": {"Rock": 0.2, "Paper": 0.5, "Scissors": 0.3},
    "Scissors": {"Rock": 0.2, "Paper": 0.1, "Scissors": 0.7}
}

if __name__ == "__main__":
    model = HMM()
    print(model)
    print()

    # Training
    start_state = "Rock"
    train_model(model, 100000, start_state, opponent_probabilities, states, False)

    print(model)
    print()

    # Testing
    results = test_model(model, 1000, start_state, opponent_probabilities, states, False)
    summarize_results(results)
    plot_results(results)

    # get_avg_results(100, 1000, model, start_state, opponent_probabilities, states, False)
    # Average results after 100 tests: 56.69
