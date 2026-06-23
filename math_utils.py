import math

def dot_product(a, b):
    """
    Compute dot product of two lists of numbers using a loop.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

def magnitude(v):
    """
    Compute the Euclidean norm (sqrt of sum of squares) using math.sqrt.
    """
    sum_sq = 0
    for x in v:
        sum_sq += x * x
    return math.sqrt(sum_sq)

def cosine_similarity(a, b):
    """
    Compute cosine similarity = dot(a,b) / (||a|| * ||b||).
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    dot = dot_product(a, b)
    mag_a = magnitude(a)
    mag_b = magnitude(b)
    # Avoid division by zero (handles zero vectors gracefully)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def mean(v):
    """
    Compute the arithmetic mean (average) of a list of numbers.
    """
    if len(v) == 0:
        raise ValueError("Cannot compute mean of empty list")
    total = 0
    for x in v:
        total += x
    return total / len(v)

def variance(v):
    """
    Compute population variance: average of squared differences from the mean.
    """
    if len(v) == 0:
        raise ValueError("Cannot compute variance of empty list")
    mu = mean(v)
    sum_sq_diff = 0
    for x in v:
        diff = x - mu
        sum_sq_diff += diff * diff
    return sum_sq_diff / len(v)