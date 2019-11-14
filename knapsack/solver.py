#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def parse_input(input_data):
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    return item_count, capacity, items

def solve_it(input_data):
    item_count, capacity, items = parse_input(input_data)
    from greedy import greedy_most_valuable, greedy_least_weight, greedy_most_dense
    from dynamic import dynamic
    if item_count == 30:
        #value, taken = greedy_most_valuable(item_count, capacity, items)
        #value, taken = greedy_least_weight(item_count, capacity, items)
        value, taken = dynamic(item_count, capacity, items)
    elif item_count == 50:
        value, taken = dynamic(item_count, capacity, items)
    elif item_count == 200:
        value, taken = dynamic(item_count, capacity, items)
    else:

        from branch import breadth_first
        value, taken = breadth_first(item_count, capacity, items)
    return prepare_output(value, taken)


def prepare_output(value, taken):
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

