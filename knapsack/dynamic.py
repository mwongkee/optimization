import numpy as np

def dynamic(item_count, capacity, items):
    # 10/10 - 99798
    # /10 - didn't try already had 10
    # 10/10 - 100236
    # /10 - crash
    # /10 - crash
    # /10 - crash
    values = np.zeros((item_count, capacity+1))
    weights = np.zeros((item_count, capacity+1))
    for i in range(item_count):
        for c in range(0, capacity+1):
            if c == 0:
                weights[i, 0] = 0
                values[i, 0] = 0
            elif i == 0:
                if c >= items[0].weight:
                    weights[0, c] = items[0].weight
                    values[0, c] = items[0].value
            elif weights[i-1, c] + items[i].weight <= c or \
                    ( c - items[i].weight >= 0 and \
                                    weights[i-1, c - items[i].weight] + items[i].weight <= c):

                if weights[i-1, c] + items[i].weight <= c:
                    v1 = values[i-1, c] + items[i].value
                else:
                    v1 = 0
                if c - items[i].weight >= 0:
                    v2 = values[i-1, c  - items[i].weight] + items[i].value
                else:
                    v2 = 0

                v3 = values[i-1, c]

                v = max(v1, v2, v3)
                values[i, c] = v
                if v1 == v:
                    weights[i, c ] = weights[i - 1, c] + items[i].weight
                elif v2 == v:
                    weights[i, c] = weights[i - 1, c - items[i].weight] + items[i].weight
                else:
                    weights[i, c] = weights[i - 1, c]
            else:
                weights[i, c] = weights[i-1, c]
                values[i, c] = values[i-1, c]

    value = values[item_count-1, capacity-1]
    taken = [0]*item_count

    c = capacity
    for i in reversed(range(1, item_count)):
        while values[i, c] == values[i, c-1]:
            c -= 1
        if values[i, c] != values[i-1, c]:
            taken[i] = 1
            c -= items[i].weight


    value = int(value)



    return value, taken