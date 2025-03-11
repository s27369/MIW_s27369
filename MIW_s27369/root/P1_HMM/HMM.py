import numpy as np
import random

balance = 0
loss= -1
win = 1

states = ["Rock", "Paper", "Scissor"]

def get_results1(my_state, opponent_state, prnt=False):
    global balance, loss, win
    if my_state == opponent_state:
        result = "Draw"

    if (
            ((my_state == "Rock") and (opponent_state == "Paper")) or
            ((my_state == "Paper") and (opponent_state == "Scissor")) or
            ((my_state == "Scissor") and (opponent_state == "Rock"))
    ):
        result = "Loss"
        balance = balance + loss
    else:
        result = "Win"
        balance = balance + win

    if prnt: print(f'{result}! Balance: {balance}')





