###########################
# 6.0002 Problem Set 1b: Space Change
# Name: sco1
# Collaborators: N/A
# Time: 1:00
# Author: charz, cdenise

#================================
# Part B: Golden Eggs
#================================

# Problem 1
def dp_make_weight(egg_weights, target_weight, memo = {}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always a egg of value 1.
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """

    # If one of our egg weights matches the target so we can skip the calculation
    if target_weight in egg_weights:
        return 1

    # Otherwise work from the ground up!
    for weight in range(1, target_weight + 1):
        if weight in egg_weights:
            # Catch here for an incorrectly (or not at all) initialized memo
            # dictionary. Because we're starting from the ground and working up,
            # if we don't have memo[1] then a KeyError will be thrown
            memo[weight] = 1
            continue  # We can also skip the calculation

        # Start with the worst case of all weight 1 eggs
        neggs = weight

        for egg in egg_weights:
            if egg > weight:
                # Skip eggs that are heavier than the weight we're calculating
                continue
            
            # See if the number of eggs needed for the current weight - the 
            # current egg weight is less than the currently calculated number of
            # eggs. If it is, then our new number of eggs is that number + 1, 
            # equivalent to adding the current egg to the previously calculated
            # weight
            if memo[weight - egg] + 1 < neggs: 
                neggs = memo[weight - egg] + 1
            
            # Store the current calculation for later use, either as a comparison
            # point or as the optimal solution for weight
            memo[weight] = neggs
    
    # Once we get here we've mapped all of the optimal egg distributions to our dictionary
    return memo[target_weight]

# EXAMPLE TESTING CODE, feel free to add more if you'd like
if __name__ == '__main__':
    egg_weights = (1, 5, 10, 25)
    n = 99
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 99")
    print("Expected ouput: 9 (3 * 25 + 2 * 10 + 4 * 1 = 99)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print()

    egg_weights = (1, 4, 5)
    n = 8
    print("Egg weights = (1, 4, 5)")
    print("n = 99")
    print("Expected ouput: 2 (2 * 4 = 99)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print()