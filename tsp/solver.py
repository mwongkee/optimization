#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from drawer import draw_solution
from naive import naive_solver
import numpy as np

Point = namedtuple("Point", ['x', 'y', 'index'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calc_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calc_total_distance(node_count, points, solution):
    obj = calc_distance(points[solution[-1]], points[solution[0]])
    for index in range(0, node_count-1):
        obj += calc_distance(points[solution[index]], points[solution[index+1]])
    return obj

def generate_output(nodeCount, solution, points):
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    return output_data

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1]), i-1))

    best_solution = None
    best_distance = None
    #   T_end = T_start * alpha ^ iter                 T_end/T_start  = alpha^iter
    #    ln (T_end/T_start) =    iter ln(alpha)
    #    iter =    ln(T_end/T_start) / ln (alpha)
    #attempts = [(1000, 1, 100)]

    if nodeCount < 2000:
        with open(r'C:\Users\mwongkee\Documents\optimization\tsp\{}.txt'.format(nodeCount), 'r') as input_data_file:
            initial = input_data_file.read()
            initial_str = initial.split('\n')[1].split(' ')
            best_solution = list(map(int, initial_str))
    else:
        best_solution = list(range(nodeCount))

    for i in range(1):
        #solution = naive_solver(nodeCount, points, 0.9999, 5, 10)  # 40416.64, #nodes 500 ->  39513
        #solution = naive_solver(nodeCount, points, 0.99999 10, 1)   #nodes 1889 ->1196847.04 0
        #solution = naive_solver(nodeCount, points, 0.99999, 10, 1, 0.00001, None)  # nodes 1889 ->458372 after 1,160,532 iterations
        #solution = naive_solver(nodeCount, points, 0.99999, 5, 10, 0.000001, initial458372)  #395518
        #solution = naive_solver(nodeCount, points, 0.99999, 100, 1, 0.000001, None)  #33810xxx
        #solution = naive_solver(nodeCount, points, 0.999999, 1, 10, 0.00001, None)  # 51 428
        #solution = naive_solver(nodeCount, points, 0.999999, 5, 100, 0.0001, None)  # 100 21294
        #solution = naive_solver(nodeCount, points, 0.99999, 20, 100, 0.0001, initial389086)  # 377137
        #solution = naive_solver(nodeCount, points, 0.99999, 9, 10, 0.00001, None)
        solution = naive_solver(nodeCount, points, 0.9, 5, 10, 0.00001, None)
        '''if nodeCount > 2500:
            n = 30
        elif nodeCount >= 200:
            n = 5
        else:
            n = 1
        solution = naive_solver(nodeCount, points, 0.9999, 1, 1) #40416.64, #nodes 500 -> T=10 39995'''
        dist = calc_total_distance(nodeCount, points, solution)
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_solution = solution

    output_data = generate_output(nodeCount, best_solution, points)

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

