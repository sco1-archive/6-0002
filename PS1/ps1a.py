###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name: sco1
# Collaborators: N/A
# Time: 1:00

from ps1_partition import get_partitions
import timeit

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    ourcows = {}
    # Loop through the lines and parse the cow name and weight
    # Assumes each line is a valid cow name/weight pair
    with open(filename, 'r') as fID:
        for tline in fID:
            # Split on the comma and add data to the dictionary
            cowname, weight = tline.split(',')
            ourcows[cowname] = int(weight)  # Convert weight to integer, also strips newline character
    
    return ourcows

# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    n_cows = len(cows)  # Get total number of cows to transport
    # Sort our cows by weight, heaviest to lightest
    sortcows = sorted(cows, key=cows.get, reverse=True)  # Returns a sorted list
    
    triplist = []  # Initialize our output list
    while len(sortcows) > 0:  # Iterate until we run out of cows to take away
        tripcows = []  # Cows on this trip
        tripweight = 0  # Current trip weight
        
        # Iterate through the available cows, adding the heaviest cow until we 
        # can't without going over the maximum trip weight
        while tripweight < limit:
            for cow in sortcows:
                if cows[cow] + tripweight <= limit:
                    tripcows.append(cow)  # Add cow to the current trip
                    tripweight += cows[cow]  # Add cow's weight to current trip weight
                    sortcows.remove(cow)  # Remove cow from the cow pool
                    break  # Force restart of the for loop because we modified what it's iterating over
            else:
                # No more cows can be added without exceeding the target weight,
                # send this trip
                break
        
        triplist.append(tripcows)  # Add the current trip to the output list of trips

    return triplist

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # Use the provided helper function to generate all unique sets of trips for
    # our given cow pool. Go through each list of trips and prune those that 
    # contain trips with weights that exceed the weight limit
    validtrips = []  # Initialize our list of valid lists of trips
    for triplist in get_partitions(cows.keys()):
        # Go through each trip of the permutation
        for trip in triplist:
            # Start a running sum of cow weights for the curren trip
            tripweight = 0
            for cow in trip:
                tripweight += cows[cow]
            if tripweight > limit:
                # This trip exceeds the trip weight limit, discard the entire
                # list of trips
                break
        else:
            # If we get here then all the trips in this list of trips do not
            # exceed the trip weight limit, append these to a list of valid lists
            validtrips.append(triplist)
    
    # Iterate through each valid list and add the number of trips to a list that
    # we can sort to find the optimal trip list, the one with the minimum number
    # of trips
    triplengths = []
    for triplist in validtrips:
        triplengths.append(len(triplist))
    
    # Sort the trip list lengths from smallest to largest
    # Rather than returning the sorted list, we instead return a list of indices
    # corresponding to the index of the item sorted list in the unsorted list.
    # e.g. if mintripidx[0] = 10, validtrips[10] is the shortest list of trips
    # http://stackoverflow.com/questions/3382352/
    mintripidx = sorted(range(len(triplengths)), key=triplengths.__getitem__)[0]

    return validtrips[mintripidx]
    
       
# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    ourcows = load_cows('.\ps1_cow_data.txt')
    weightlimit = 10

    # Using timeit rather than time to avoid system clock resolution issues
    start1 = timeit.default_timer()
    trips = greedy_cow_transport(ourcows, weightlimit)
    end1 = timeit.default_timer()

    start2 = timeit.default_timer()
    trips2 = brute_force_cow_transport(ourcows, weightlimit)
    end2 = timeit.default_timer()

    print('Cows used:')
    print(ourcows)
    print('Weight Limit: %i' % weightlimit)
    print('=============')
    print('Greedy Algorithm Trips:')
    print(trips)
    print('Greedy Algorithm Time: %f seconds' % (end1 - start1))
    print('Number of trips: %i' % len(trips))
    print('=============')
    print('Brute Force Algorithm Trips:')
    print(trips2)
    print('Brute Force Algorithm Time: %f seconds' % (end2 - start2))
    print('Number of trips: %i' % len(trips2))
    print('=============')
