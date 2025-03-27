class HMM:
    def __init__(self, states:list):
        for state in states:
            self.probabilities[state] = {x:1/len(states) for x in states}
            #opponent actions
            self.observations[state] = {x:0 for x in states}


