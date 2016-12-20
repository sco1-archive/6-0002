def solve(s):
    """ 
    s: positive integer, what the sum should add up to
    Solves the following optimization problem:
        x1 + x2 + x3 + x4 is minimized 
        subject to the constraint x1*25 + x2*10 + x3*5 + x4 = s
        and that x1, x2, x3, x4 are non-negative integers.
    Returns a list of the coefficients x1, x2, x3, x4 in that order
    """

    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0

    # This is an absolutely horrendous approach
    while True:
        if s - 25 >= 0:
            x1 += 1
            s = s - 25
        else:
            break
    
    while True:
        if s - 10 >= 0:
            x2 += 1
            s = s - 10
        else:
            break

    while True:
        if s - 5 >= 0:
            x3 += 1
            s = s - 5
        else:
            break

    x4 = s

    return [x1, x2, x3, x4]

print(solve(5))
print(solve(10))
print(solve(11))
print(solve(12))
print(solve(13))
print(solve(14))
print(solve(15))
print(solve(4))