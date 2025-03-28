import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toolbox as tob


def get_state_to_index_mapping(states):
    """
    Creates a mapping from states to their indices.

    Args:
        states (list): List of states.

    Returns:
        dict: Mapping of states to indices.
    """
    return {state: idx for idx, state in enumerate(states)}


def simulate_sequence(states, prob_matrix, turns):
    """
    Simulates a sequence of states based on a probability matrix.

    Args:
        states (list): List of states.
        prob_matrix (np.ndarray): Transition probability matrix.
        turns (int): Number of turns to simulate.

    Returns:
        list: Simulated sequence of states.
    """
    state_to_index = get_state_to_index_mapping(states)
    current_state = np.random.choice(states)
    sequence = [current_state]

    for _ in range(turns - 1):
        current_index = state_to_index[current_state]
        next_state = np.random.choice(states, p=prob_matrix[current_index])
        sequence.append(next_state)
        current_state = next_state

    return sequence


def calculate_occurrence_matrix(sequence, states):
    """
    Calculates the occurrence matrix from a sequence of states.

    Args:
        sequence (list): Sequence of states.
        states (list): List of all possible states.

    Returns:
        np.ndarray: Occurrence matrix.
    """
    state_to_index = get_state_to_index_mapping(states)
    occurrence_matrix = np.zeros((len(states), len(states)), dtype=int)

    for i in range(1, len(sequence)):
        prev_index = state_to_index[sequence[i - 1]]
        current_index = state_to_index[sequence[i]]
        occurrence_matrix[prev_index][current_index] += 1

    return occurrence_matrix


def decide_winner(prediction, actual, state_matrix):
    """
    Determines the outcome of a game based on the prediction and actual state.

    Args:
        prediction (str): Predicted state.
        actual (str): Actual state.
        state_matrix (list): List of all possible states in order.

    Returns:
        int: 1 for win, 0 for tie, -1 for loss.
    """
    state_to_index = get_state_to_index_mapping(state_matrix)
    pred_index = state_to_index[prediction]
    actual_index = state_to_index[actual]

    # Calculate the difference in indices
    diff = (actual_index - pred_index) % len(state_matrix)

    if diff == 1:  # n+1 (win)
        return 1
    elif diff == len(state_matrix) - 1:  # n-1 (loss)
        return -1
    else:  # Any other state (tie)
        return 0


def generate_probability_matrix(states):
    """
    Generates a random probability matrix for the given states.

    Args:
        states (list): List of states.

    Returns:
        np.ndarray: Randomly generated probability matrix.
    """
    return np.array(
        [tob.generate_random_array(len(states), decimals=3) for _ in states])


def plot_rewards(rewards_history):
    """
    Plots the rewards over time.

    Args:
        rewards_history (list): List of cumulative rewards over time.
    """
    plt.plot(rewards_history, label="Rewards")
    plt.xlabel("Turns")
    plt.ylabel("Rewards")
    plt.title("Rewards Over Time in HMM Simulation")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    """
    Main function to simulate the game and evaluate the HMM model.
    """
    # Input and validation
    state_matrix = input("Enter states (comma-separated): ").split(",")
    state_matrix = [state.strip() for state in state_matrix if state.strip()]
    if len(state_matrix) < 2:
        print("Error: At least two states are required.")
        return

    prev_games_number = int(
        input("Enter the number of previous games to simulate: "))
    turns_to_play = int(
        input("Enter the number of turns to test the program: "))

    # Generate probability matrix and simulate initial sequence
    probability_matrix = generate_probability_matrix(state_matrix)
    # probability_matrix = [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    
    
    sequence = simulate_sequence(state_matrix, probability_matrix,
                                 prev_games_number)

    # Calculate occurrence matrix
    occurrence_matrix = calculate_occurrence_matrix(sequence, state_matrix)

    # Display matrices
    print("\nOccurrence Matrix:")
    print(
        pd.DataFrame(occurrence_matrix,
                     index=state_matrix,
                     columns=state_matrix))
    print("\nProbability Matrix:")
    print(
        pd.DataFrame(probability_matrix,
                     index=state_matrix,
                     columns=state_matrix))

    # Initialize variables for simulation
    state_to_index = get_state_to_index_mapping(state_matrix)
    rules = {
        state_matrix[i]: state_matrix[(i + 1) % len(state_matrix)]
        for i in range(len(state_matrix))
    }
    rewards, ties, wins, losses = 0, 0, 0, 0
    rewards_history = []

    # Simulate the game
    for i in range(turns_to_play):
        current_state = sequence[-1]
        row_sum = occurrence_matrix[state_to_index[current_state]].sum()

        if row_sum == 0:
            prediction = np.random.choice(state_matrix)
        else:
            prediction = np.random.choice(
                state_matrix,
                p=occurrence_matrix[state_to_index[current_state]] / row_sum)

        next_state = np.random.choice(
            state_matrix,
            p=occurrence_matrix[state_to_index[current_state]] / row_sum)

        occurrence_matrix[state_to_index[current_state]][
            state_to_index[next_state]] += 1
        sequence.append(next_state)

        result = decide_winner(prediction, next_state, rules)
        
        if result == -1:
            losses += 1
        elif result == 0:
            ties += 1
        else:
            wins += 1
            
        rewards += result
        rewards_history.append(rewards)

        if i % 100 == 0:
            print(
                f"{i}th iteration, state: {next_state}, rewards: {rewards}, wins: {wins}, ties: {ties}, losses: {losses}"
            )

    print(f"\nTotal rewards: {float(rewards)/len(rewards_history)*100}")

    # Plot rewards
    plot_rewards(rewards_history)


if __name__ == "__main__":
    main()
