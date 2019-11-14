#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict, namedtuple

# 1 - 10/10  6  7(7)
# 2 - 3/10  21 -> 20
# 3 - 7/10 21 -> 20
# 4 - 3/10 96 -> 113,104,100
# 5 - 3/10 19 -> 20 (need 18)
# 6 - 7/10 121 -> 148 (3)after more relaxation

import timer
import numpy as np
import random


class Vertex(object):
    def __init__(self, index):
        self.index = index
        self.neighbours = []
        self.iteration = None
        self.color = None
        self.neighbour_count = 0
        self.possible_choices = None # this is only used at the decision point and not dynamically updated
        self.choices_made = None # this is only used at the decision point and not dynamically updated

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)
        self.neighbour_count += 1

def get_constraints(vertices, colors):
    constraints = {}
    for vert in vertices.values():
        neighbour_colors = set([])
        for conn in vert.neighbours:
            if conn in colors:
                neighbour_colors.add(colors[conn])
        constraints[vert.index] = neighbour_colors
    return constraints

def get_possible_colors(vert, colors):
    # neighbours = constraints.get(vert.index)

    neighbour_colors = set([])
    for neighbour in vert.neighbours:
        neighbour_color = colors.get(neighbour)
        if neighbour_color is not None:
            neighbour_colors.add(neighbour_color)


    max_color = max(colors.values()) if colors else -1 # so possible colors will be 0
    possible_colors = set(range(max_color+2))
    possible_colors = possible_colors - neighbour_colors
    return possible_colors


def assign_color(vert, colors, constraints, choices, iteration):
    if vert.possible_choices is None:
        possible_colors = get_possible_colors(vert, colors)
        vert.possible_choices = list(possible_colors)
        vert.choices_made = []

    color_choice = vert.possible_choices.pop(0)
    vert.choices_made.append(color_choice)
    vert.color = color_choice
    colors[vert.index] = color_choice
    vert.iteration = iteration
    if vert.possible_choices:
        choices.append(vert)



def get_top_neighbour_counts(vertices, top):
    potential_vertices = list(filter(lambda v: v.color is None, vertices.values()))
    if not potential_vertices:
        return None
    n = min(len(potential_vertices), top)
    sorted_ind = np.array(list(v.neighbour_count for v in potential_vertices)).argsort()
    sorted_vert = [potential_vertices[i] for i in sorted_ind]
    cutoff = sorted_vert[-1*n].neighbour_count
    return [v for v in sorted_vert if v.neighbour_count >= cutoff]

def get_top_constrained(vert_values, top, constraints):
    if not vert_values:
        return None
    n = min(len(vert_values), top)
    constraint_counts = list(len(constraints[v.index]) for v in vert_values)
    sorted_ind = np.array(constraint_counts).argsort()
    sorted_vert = [vert_values[i] for i in sorted_ind]
    sorted_counts = [constraint_counts[i] for i in sorted_ind]
    cutoff = constraint_counts[-1 * n]
    return [sorted_vert[i] for i in range(len(sorted_vert)) if sorted_counts[i] <= cutoff]


def get_potential_next_vertex(counts, vertices, constraints, top_counts, top_constrained):
    top_counts = get_top_neighbour_counts(vertices, top_counts)
    top_constrained = get_top_constrained(top_counts, top_constrained, constraints)
    return top_constrained

def get_next_vertex(counts, vertices, constraints, top_counts=1, top_constrained=1):
    top_constrained = get_potential_next_vertex(counts, vertices, constraints, top_counts, top_constrained)
    return top_constrained[random.randint(0,len(top_constrained)-1)] if top_constrained else None


def clean_vertex(vert, colors):
    vert.color = None
    vert.possible_choices = None
    vert.choices_made = None
    vert.iteration = None
    del colors[vert.index]

def pop_choice(vertices, choices, colors):
    #print('popping choice')

    last_choice = choices.pop(-1)
    last_iteration = last_choice.iteration
    for vert in vertices.values():
        if vert.iteration is not None and vert.iteration > last_iteration:
            clean_vertex(vert, colors)
    if last_choice.possible_choices:
        num_colors = max(colors.values()) + 1
        #print(
        #    ['num_colors:{} {}:{}/{}'.format(num_colors, c.index, c.choices_made, c.possible_choices) for c in choices])
        return last_choice
    else:
        num_colors = max(colors.values()) + 1
        #print(
        #    ['num_colors:{} {}:{}/{}'.format(num_colors, c.index, c.choices_made, c.possible_choices) for c in choices])
        return None
    #print(['num_colors:{} {}:{}/{}'.format(num_colors, c.index, c.choices_made, c.possible_choices) for c in choices])

def populate_vertices(edges):
    vertices = {}
    for edge in edges:
        p1, p2 = edge
        v1 = vertices.get(p1)
        v2 = vertices.get(p2)
        if not v1:
            v1 = Vertex(p1)
            vertices[p1] = v1
        v1.add_neighbour(p2)

        if not v2:
            v2 = Vertex(p2)
            vertices[p2] = v2
        v2.add_neighbour(p1)
    return vertices

def attempt(edges, node_count, max_iters, previous_best, top_counts=1, top_constrained=1, scenario_colors=None, starting_points=None):

    vertices = populate_vertices(edges)



    #connections = defaultdict(list)

    counts = dict((v.index, v.neighbour_count) for v in vertices.values())
    TOP_COUNTS = 4
    if scenario_colors is None:
        starting_points = get_top_neighbour_counts(vertices, TOP_COUNTS)
        scenario_colors = [(0,0,0,0), (0,0,1,0), (0,0,2,0), (0,1,0,0), (0,1,1,0), (0,1,2,0),
                           (0, 0, 0,1), (0, 0, 1,1), (0, 0, 2,1), (0, 1, 0,1), (0, 1, 1,1), (0, 1, 2,1),
                           (0, 0, 0,2), (0, 0, 1,2), (0, 0, 2,2), (0, 1, 0,2), (0, 1, 1,2), (0, 1, 2,2),
                           (0, 0, 0,3), (0, 0, 1,3), (0, 0, 2,3), (0, 1, 0,3), (0, 1, 1,3), (0, 1, 2,3)]
    best_result = None
    best_colors = None
    best_scenario_colors = None
    best_scenario_starting_points = None
    for scenario in scenario_colors:

        feasible = True
        colors = {}

        vertices = populate_vertices(edges) # reinit
        for i in range(TOP_COUNTS):
            ind = starting_points[i].index
            color = scenario[i]
            constraints = get_constraints(vertices, colors)
            if color in constraints[ind]:
                feasible = False # infeasible scenario
            colors[ind] = color
            vertices[ind].color = color
        if feasible:
            previous_best = best_result if best_result else previous_best
            num, result = do_sim(node_count, vertices, colors, max_iters, previous_best, counts, top_counts, top_constrained)
            print("attempted scenario:{} colors:{}".format(scenario, num))
            if num is not None:
                print("new best scenario:{}:{}".format(scenario, num))
                best_result = num
                best_colors = result
                best_scenario_colors = scenario
                best_scenario_starting_points = starting_points
        else:
            print("scenario: {} infeasible".format(scenario))
    return best_result, best_colors, best_scenario_colors, best_scenario_starting_points


def do_sim(node_count, vertices, colors, max_iters, previous_best, counts, top_counts=1, top_constrained=1):
    print("do_sim: max_iters:{} previous_best:{} top_counts:{}, top_constrained:{}".format(max_iters, previous_best, top_counts, top_constrained))
    best_result = previous_best if previous_best else None
    best_colors = None

    import time
    start_time = time.time()
    choices = []
    iteration = 0
    prev_choice = None
    THRESHOLD = 60000 + node_count * 2
    while True:
        if iteration > max_iters or (time.time() - start_time) > THRESHOLD:
            if best_colors is None:
                return None, None
            else:
                return best_result, best_colors
        iteration += 1
        constraints = get_constraints(vertices, colors)

        if prev_choice:
            vert = prev_choice
            prev_choice = None
        else:
            vert = get_next_vertex(counts, vertices, constraints, top_counts, top_constrained)
        if vert is None:
            if not choices:
                break
            prev_choice = pop_choice(vertices, choices, colors)
            continue

        assign_color(vert, colors, constraints, choices, iteration)
        num_nodes = len(colors.keys())
        num_colors = max(colors.values()) + 1
        #print('best:{} num_nodes:{} num colors:{}'.format(best_result, num_nodes, num_colors))
        if len(colors) == node_count:
            if best_result is None or num_colors < best_result:
                print("new best results:" + str(num_colors))
                print(colors)
                best_result = num_colors
                best_colors = colors.copy()
                prev_choice = pop_choice(vertices, choices, colors)
        elif num_colors is not None and best_result is not None and num_colors >= best_result:
            if not choices:
                break
            if max(colors.values()) + 1 >= best_result:
                #while max(colors.values()) + 1 >= 10:
                    if not choices:
                        break
                    prev_choice = pop_choice(vertices, choices, colors)
            continue
    return best_result, best_colors



def solve_it(input_data):
    # Modify this code to run your optimization algorithm


    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    best_colors = None
    previous_best = None
    best_scenario_colors = None
    max_iters = node_count * 50


    scenario_colors = None
    scenario_starting_points = None

    stop_criteria = {
        50: 6,
        70: 20,
        100: 16,
        250: 78,
        500: 16,
        1000: 100
    }

    for i in range(50):
        import math
        max_iters *= 2
        num, result, colors, starting_points = attempt(edges, node_count, max_iters, previous_best, np.mod(i,5)+1, np.mod(i,5)+1)
        if num is not None:
            previous_best = num
            best_colors = result
            scenario_colors = colors
            scenario_starting_points = starting_points

            '''fmYLC, ./data/gc_50_3, solver.py, Coloring Problem 1
IkKpq, ./data/gc_70_7, solver.py, Coloring Problem 2
pZOjO, ./data/gc_100_5, solver.py, Coloring Problem 3
XDQ31, ./data/gc_250_9, solver.py, Coloring Problem 4
w7hAO, ./data/gc_500_1, solver.py, Coloring Problem 5
tthbm, ./data/gc_1000_5, solver.py, Coloring Problem 6
'''
            if num == stop_criteria[node_count]:
                break

    max_iters = node_count * 50
    for i in range(0):
        max_iters *= 1.1
        num, result, _, _ = attempt(edges, node_count, max_iters, previous_best,  i+1, i+1, [scenario_colors], scenario_starting_points)
        if num is not None:
            previous_best = num
            best_colors = result
            best_scenario_colors



    solution = []
    for i in range(node_count):
        solution.append(best_colors[i])

    value = max(solution) + 1

    # build a trivial solution
    # every node has its own color
    #solution = range(0, node_count)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

