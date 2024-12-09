import numpy as np

def fit_01_knapsack(solution, items, W):
    total_value = 0
    total_weight = 0
    for i, include in enumerate(solution):
        if include:
            total_value += items[i][0]
            total_weight += items[i][1]
    return -total_value if total_weight <= W else 0

def fit_bounded_knapsack(solution, items, W):
    total_value = 0
    total_weight = 0
    for i, count in enumerate(solution):
        if count > items[i][2]:  # Check if count exceeds max_count
            return 0
        total_value += items[i][0] * count
        total_weight += items[i][1] * count
    return -total_value if total_weight <= W else 0

def fit_unbounded_knapsack(solution, items, W):
    total_value = 0
    total_weight = 0
    for i, count in enumerate(solution):
        total_value += items[i][0] * count
        total_weight += items[i][1] * count
    return -total_value if total_weight <= W else 0

def print_knapsack_solution(solution, items):
    total_value = 0
    total_weight = 0
    for i, count in enumerate(solution):
        if count > 0:
            value, weight = items[i][0] * count, items[i][1] * count
            print(f"Item {i+1}: count={count}, value={value}, weight={weight}")
            total_value += value
            total_weight += weight
    print(f"Total value: {total_value}")
    print(f"Total weight: {total_weight}")