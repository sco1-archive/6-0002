import random

# helper function
def getMeanAndStd(X):
    mean = sum(X)/float(len(X))
    tot = 0.0
    for x in X:
        tot += (x - mean)**2
    std = (tot/len(X))**0.5
    return mean, std  
  
def guessfood_sim(num_trials, probs, cost, get):
    """
    num_trials: integer, number of trials to run
    probs: list of probabilities of guessing correctly for 
           the ith food, in each trial
    cost: float, how much to pay for each food guess
    get: float, how much you get for a correct guess
    
    Runs a Monte Carlo simulation, 'num_trials' times. In each trial 
    you guess what each food is, the ith food has 'prob[i]' probability 
    to get it right. For every food you guess, you pay 'cost' dollars.
    If you guess correctly, you get 'get' dollars. 
    
    Returns: a tuple of the mean and standard deviation over 
    'num_trials' trials of the net money earned 
    when making len(probs) guesses
    """
    dollarydoos_all = []
    for trial in range(0, num_trials):
        dollarydoos = 0
        for food_prob in probs:
            dollarydoos -= cost
            if random.random() < food_prob:
                dollarydoos += get
        else:
            dollarydoos_all.append(dollarydoos)
    
    return getMeanAndStd(dollarydoos_all)

print(guessfood_sim(100, [0, 1, 1], 0.5, 1))
print(guessfood_sim(100, [0, 1, 0], 1, 0.5))
print(guessfood_sim(3000, [0.25, 0.5, 0.75], 1.5, 2.5))
print(guessfood_sim(3000, [0.5, 0.4, 0.9], 2.2, 10.3))
print(guessfood_sim(3000, [0.3, 0.001, 0.1], 3, 3))
print(guessfood_sim(3000, [0.1, 0.99, 0.4], 1, 1))