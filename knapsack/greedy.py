
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def greedy_most_valuable(item_count, capacity, items):
    # 3/10 - 90000
    # 10/10 - 142156
    # 3/10 - 90001
    # 7/10 - 3966825
    # 3/10 - 107768
    # 3/10 - 1094968
    weight = 0
    value = 0
    taken = [0] * item_count
    items = sorted(items, key=lambda x: x.value, reverse=True)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return value, taken


def greedy_least_weight(item_count, capacity, items):
    # 7/10 - 99045
    # 3/10 - 132044
    # 3/10 - 99090
    # 3/10 - 3879439
    # 3/10 - 99488
    # 3/10 - 994412
    weight = 0
    value = 0
    taken = [0] * item_count
    items = sorted(items, key=lambda x: x.weight)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return value, taken

def greedy_most_dense(item_count, capacity, items):
    # 7/10 - 99084
    # 3/10 - 135269
    # 3/10 - 90001
    # 3/10 - 3900775
    # 3/10 - 90045
    # 3/10 - 900124 
    weight = 0
    value = 0
    taken = [0] * item_count
    items = sorted(items, key=lambda x: float(x.value)/float(x.weight))

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return value, taken