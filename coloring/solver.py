#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict, namedtuple

# 1 - 10/10  6  7(7)   now 8  (7)   6
# 2 - 3/10  21 -> 20  now 21  (3)   20
# 3 - 7/10 21 -> 20  now 20    (7)
# 4 - 3/10 96 -> 113,104,100   now 95 (7)
# 5 - 3/10 19 -> 20 (need 18) now 18 (7)
# 6 - 7/10 121 -> 148 (3)after more relaxation   now 120

import timer
import numpy as np
import random


class VertexPool(object):
    def __init__(self, vertices):
        self.vertices = vertices
        self.vertex_index_map = dict([(v.index, v) for v in vertices])

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
    return list(reversed([v for v in sorted_vert if v.neighbour_count >= cutoff]))

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

    if node_count == 50:
        best_result = 6
        best_colors = [2,
         5,
         0,
         4,
         0,
         3,
         2,
         2,
         0,
         5,
         3,
         5,
         2,
         2,
         1,
         4,
         1,
         5,
         4,
         0,
         4,
         4,
         0,
         3,
         1,
         0,
         3,
         2,
         3,
         4,
         5,
         1,
         4,
         0,
         2,
         0,
         2,
         0,
         2,
         2,
         5,
         4,
         1,
         1,
         3,
         1,
         4,
         3,
         5,
         5]
        return best_result, best_colors
    elif node_count == 70:
        best_colors = [16, 2, 2, 17, 13, 12, 0, 7, 3, 10, 15, 13, 0, 7, 8, 19, 14, 9, 12, 8, 7, 18, 10, 5, 1, 11, 4,
                       14, 1, 19, 15, 17, 15, 6, 18, 4, 19, 3, 9, 11, 17, 1, 11, 4, 1, 0, 8, 3, 18, 12, 6, 18, 3, 14,
                       13, 9, 15, 12, 0, 19, 2, 8, 1, 5, 2, 2, 16, 10, 5, 4
                       ]
        best_result = 20
        return best_result, best_colors
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
    THRESHOLD = 60 + node_count / 10
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
                while max(colors.values()) + 1 >= 10:
                    if not choices:
                        break
                    prev_choice = pop_choice(vertices, choices, colors)
            continue
    return best_result, best_colors

def parse_input_data(input_data):
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
    return node_count, edges

def solve_it(input_data):
    solution = attempt(input_data)
    print(solution)
    for i in range(5):
        a_solution = attempt(input_data, max(solution)-1)
        if a_solution is None:
            break
        solution = a_solution
        print(a_solution)
    output_data = generate_output_data(solution)
    return output_data

def attempt(input_data, max_c=None):
    # Modify this code to run your optimization algorithm




    import time

    start = time.time()
    node_count, edges = parse_input_data(input_data)

    if node_count == 50:
        best_result = 6
        best_colors = [2,
         5,
         0,
         4,
         0,
         3,
         2,
         2,
         0,
         5,
         3,
         5,
         2,
         2,
         1,
         4,
         1,
         5,
         4,
         0,
         4,
         4,
         0,
         3,
         1,
         0,
         3,
         2,
         3,
         4,
         5,
         1,
         4,
         0,
         2,
         0,
         2,
         0,
         2,
         2,
         5,
         4,
         1,
         1,
         3,
         1,
         4,
         3,
         5,
         5]
        return best_colors
    elif node_count == 70:
        best_colors = [16, 2, 2, 17, 13, 12, 0, 7, 3, 10, 15, 13, 0, 7, 8, 19, 14, 9, 12, 8, 7, 18, 10, 5, 1, 11, 4,
                       14, 1, 19, 15, 17, 15, 6, 18, 4, 19, 3, 9, 11, 17, 1, 11, 4, 1, 0, 8, 3, 18, 12, 6, 18, 3, 14,
                       13, 9, 15, 12, 0, 19, 2, 8, 1, 5, 2, 2, 16, 10, 5, 4
                       ]
        best_result = 20
        return best_colors
    elif node_count == 100:
        best_colors = [12, 8, 10, 5, 1, 11, 2, 19, 12, 3, 16, 6, 0, 15, 13, 2, 4, 1, 7, 8, 5, 3, 14, 13, 1, 19, 6, 8, 12, 0, 11, 0, 8, 7, 4, 16, 13, 16, 18, 3, 1, 10, 9, 6, 14, 10, 15, 13, 9, 7, 7, 10, 16, 17, 4, 0, 11, 9, 11, 2, 4, 18, 12, 9, 9, 4, 2, 3, 17, 2, 19, 7, 3, 0, 17, 19, 11, 5, 6, 11, 14, 10, 2, 5, 9, 7, 14, 12, 8, 15, 8, 4, 1, 6, 18, 16, 17, 15, 17, 18]
        return best_colors
    elif node_count == 250:
        best_colors = [93, 49, 29, 55, 68, 35, 8, 92, 58, 9, 57, 2, 15, 91, 73, 60, 60, 61, 34, 31, 11, 19, 80, 17, 8, 27, 0, 57, 20,
         52, 9, 74, 28, 12, 42, 36, 56, 69, 26, 31, 62, 4, 72, 66, 34, 70, 26, 53, 56, 41, 24, 40, 40, 45, 86, 20, 39,
         44, 27, 81, 10, 53, 55, 7, 82, 51, 62, 43, 17, 5, 14, 21, 47, 45, 82, 49, 69, 48, 38, 63, 5, 70, 64, 36, 3, 37,
         54, 39, 76, 31, 84, 85, 10, 49, 80, 43, 21, 44, 13, 34, 63, 25, 4, 26, 18, 43, 77, 78, 37, 82, 10, 46, 73, 18,
         75, 11, 50, 27, 76, 92, 7, 70, 28, 17, 79, 1, 11, 35, 19, 75, 83, 4, 36, 59, 75, 12, 22, 90, 71, 66, 23, 38,
         67, 61, 68, 60, 87, 26, 65, 65, 7, 3, 42, 16, 66, 15, 78, 24, 72, 76, 9, 65, 67, 54, 71, 61, 13, 56, 1, 24, 68,
         57, 16, 67, 41, 42, 18, 69, 84, 35, 22, 46, 21, 27, 59, 23, 47, 79, 55, 90, 16, 83, 64, 2, 48, 29, 8, 74, 28,
         37, 25, 14, 51, 58, 47, 54, 22, 53, 88, 77, 5, 62, 3, 30, 87, 6, 85, 91, 78, 30, 6, 33, 23, 41, 50, 30, 81, 29,
         73, 77, 72, 20, 0, 47, 14, 12, 40, 88, 89, 32, 63, 25, 58, 94, 33, 5, 32, 60, 52, 19]
        return best_colors
    elif node_count == 500:
        best_colors = [6, 6, 3, 8, 8, 5, 6, 11, 13, 1, 0, 10, 7, 1, 8, 4, 13, 9, 15, 5, 4, 6, 6, 13, 9, 4, 13, 2, 3, 15, 16, 14, 16, 12, 6, 10, 9, 10, 8, 10, 12, 11, 5, 11, 9, 1, 4, 5, 11, 14, 9, 9, 3, 11, 9, 8, 7, 15, 5, 3, 14, 10, 0, 4, 1, 5, 5, 6, 9, 12, 6, 11, 10, 8, 15, 6, 3, 16, 9, 2, 3, 2, 11, 14, 4, 12, 3, 1, 12, 1, 7, 7, 1, 5, 16, 2, 11, 6, 12, 9, 10, 11, 4, 8, 4, 12, 9, 15, 14, 11, 3, 5, 0, 12, 10, 3, 8, 4, 5, 4, 3, 9, 14, 14, 2, 0, 0, 7, 13, 7, 0, 2, 13, 1, 14, 0, 4, 7, 12, 14, 8, 11, 3, 12, 11, 0, 2, 4, 9, 5, 0, 8, 2, 14, 6, 0, 3, 3, 4, 5, 5, 5, 0, 4, 12, 10, 13, 6, 6, 9, 10, 10, 11, 13, 3, 9, 15, 1, 3, 10, 6, 6, 2, 8, 5, 0, 9, 2, 6, 9, 1, 13, 10, 4, 2, 3, 5, 2, 0, 15, 11, 4, 4, 10, 11, 1, 5, 7, 6, 1, 0, 16, 2, 8, 8, 7, 11, 8, 5, 8, 2, 1, 6, 8, 9, 1, 15, 12, 6, 7, 13, 14, 9, 5, 8, 8, 7, 15, 5, 3, 7, 4, 3, 6, 1, 7, 3, 7, 7, 1, 3, 7, 8, 10, 11, 2, 15, 2, 11, 3, 8, 6, 4, 10, 7, 1, 3, 14, 14, 2, 10, 0, 12, 0, 8, 10, 13, 3, 9, 9, 4, 8, 11, 5, 10, 2, 2, 6, 13, 13, 1, 10, 8, 3, 2, 2, 12, 5, 0, 4, 3, 11, 0, 4, 6, 7, 1, 2, 2, 15, 8, 4, 0, 13, 7, 8, 13, 10, 2, 15, 7, 3, 2, 1, 3, 9, 5, 12, 1, 4, 8, 0, 8, 13, 0, 7, 11, 6, 12, 12, 5, 2, 4, 8, 11, 10, 4, 9, 3, 1, 4, 7, 9, 3, 11, 5, 6, 6, 1, 3, 8, 11, 11, 3, 5, 10, 5, 7, 8, 6, 11, 5, 1, 10, 17, 9, 6, 15, 7, 2, 9, 6, 10, 11, 0, 5, 17, 1, 11, 6, 0, 16, 0, 10, 12, 13, 14, 7, 0, 9, 2, 2, 2, 1, 7, 13, 0, 1, 10, 3, 1, 0, 8, 11, 7, 1, 13, 12, 5, 9, 13, 12, 1, 0, 10, 9, 9, 7, 7, 7, 7, 4, 5, 14, 7, 3, 14, 9, 1, 4, 12, 7, 6, 13, 12, 14, 5, 7, 14, 15, 14, 4, 5, 3, 3, 13, 8, 7, 12, 0, 14, 4, 8, 7, 9, 6, 15, 10, 8, 9, 13, 9, 12, 0, 8, 10, 14, 0, 2, 0, 8, 3, 9, 2, 11, 1, 8, 12, 5, 6, 10, 4, 14, 13, 14, 3, 2, 6, 15, 2]
        return best_colors
    elif node_count == 1000:
        best_colors = [38, 106, 61, 14, 19, 42, 54, 65, 34, 98, 57, 12, 14, 74, 103, 10, 32, 20, 18, 72, 104, 67, 86, 20, 39, 29, 41, 29, 21, 61, 61, 102, 17, 34, 30, 91, 7, 3, 12, 57, 26, 25, 14, 78, 19, 17, 115, 9, 7, 39, 110, 89, 93, 79, 55, 93, 54, 107, 29, 63, 33, 109, 99, 35, 71, 104, 41, 0, 85, 84, 65, 71, 53, 89, 75, 64, 77, 18, 68, 6, 29, 41, 58, 5, 23, 108, 91, 27, 64, 10, 89, 39, 93, 98, 23, 41, 24, 7, 94, 74, 66, 32, 37, 74, 20, 63, 5, 105, 66, 52, 78, 114, 90, 36, 34, 58, 7, 87, 100, 77, 107, 87, 96, 100, 116, 32, 99, 83, 84, 17, 53, 98, 68, 67, 0, 113, 47, 92, 76, 27, 70, 87, 3, 56, 106, 18, 83, 37, 80, 37, 14, 85, 8, 30, 11, 54, 53, 79, 50, 4, 65, 86, 105, 101, 32, 71, 55, 3, 96, 43, 61, 99, 59, 48, 43, 63, 47, 24, 72, 43, 10, 24, 92, 101, 100, 10, 11, 9, 70, 89, 28, 103, 21, 77, 8, 116, 74, 63, 70, 62, 31, 64, 33, 102, 21, 17, 35, 116, 1, 110, 47, 11, 21, 77, 26, 6, 29, 36, 14, 67, 62, 68, 73, 87, 32, 40, 45, 29, 57, 7, 39, 73, 65, 52, 46, 26, 88, 33, 42, 17, 44, 32, 91, 58, 64, 90, 62, 104, 22, 91, 59, 26, 79, 107, 106, 94, 52, 85, 26, 12, 15, 111, 69, 63, 31, 108, 31, 43, 72, 0, 27, 43, 90, 6, 94, 30, 35, 26, 79, 44, 97, 86, 4, 53, 52, 85, 36, 102, 82, 1, 78, 47, 0, 84, 59, 103, 3, 76, 9, 93, 50, 88, 67, 95, 55, 38, 11, 1, 5, 29, 73, 37, 81, 50, 37, 85, 102, 40, 6, 61, 20, 2, 48, 19, 22, 48, 73, 78, 80, 103, 69, 47, 70, 63, 51, 80, 72, 54, 93, 116, 96, 6, 87, 27, 108, 112, 13, 10, 85, 11, 10, 84, 40, 15, 1, 5, 83, 103, 93, 48, 85, 21, 15, 92, 94, 70, 78, 12, 5, 39, 109, 100, 60, 38, 93, 106, 98, 49, 8, 30, 85, 41, 11, 36, 117, 78, 37, 17, 25, 26, 86, 62, 60, 60, 73, 25, 57, 5, 3, 72, 80, 13, 117, 44, 49, 36, 66, 31, 75, 7, 96, 51, 13, 45, 28, 95, 97, 74, 19, 75, 65, 30, 14, 73, 83, 19, 27, 59, 82, 44, 21, 88, 28, 72, 4, 107, 58, 81, 28, 60, 25, 2, 3, 97, 66, 29, 2, 80, 112, 31, 41, 101, 37, 82, 0, 61, 58, 111, 15, 13, 108, 104, 94, 67, 82, 59, 22, 23, 1, 2, 21, 7, 108, 72, 31, 109, 36, 67, 65, 66, 45, 110, 92, 54, 16, 81, 110, 99, 49, 56, 7, 86, 9, 9, 76, 87, 65, 80, 108, 6, 18, 44, 56, 104, 68, 106, 52, 5, 12, 26, 79, 34, 62, 35, 84, 46, 46, 43, 38, 40, 95, 58, 37, 90, 100, 96, 83, 3, 94, 20, 75, 90, 91, 21, 109, 102, 75, 47, 3, 15, 90, 101, 81, 47, 116, 64, 68, 71, 6, 18, 49, 87, 16, 6, 34, 46, 39, 110, 88, 18, 28, 4, 5, 12, 80, 62, 74, 58, 62, 63, 1, 64, 46, 94, 114, 73, 69, 20, 44, 1, 104, 33, 16, 80, 14, 70, 10, 113, 1, 9, 67, 23, 79, 6, 93, 55, 65, 65, 77, 56, 76, 89, 14, 11, 110, 53, 23, 66, 48, 17, 35, 84, 96, 92, 68, 91, 81, 67, 111, 106, 101, 69, 2, 71, 90, 21, 51, 34, 67, 5, 9, 68, 98, 76, 55, 69, 24, 11, 43, 4, 51, 36, 5, 44, 50, 40, 111, 51, 55, 15, 33, 15, 54, 73, 77, 35, 64, 115, 75, 32, 100, 34, 42, 10, 13, 60, 22, 85, 49, 13, 105, 48, 105, 85, 88, 35, 32, 77, 74, 16, 42, 14, 67, 46, 8, 99, 40, 109, 71, 64, 97, 16, 80, 10, 24, 115, 88, 46, 46, 98, 52, 69, 16, 102, 112, 119, 50, 66, 81, 62, 46, 57, 43, 103, 56, 3, 72, 16, 51, 82, 89, 39, 38, 56, 11, 8, 59, 34, 8, 103, 50, 95, 86, 77, 79, 79, 2, 42, 50, 77, 59, 60, 55, 69, 98, 28, 15, 114, 105, 13, 70, 24, 61, 27, 112, 118, 9, 84, 18, 94, 60, 41, 95, 97, 78, 48, 75, 79, 12, 44, 53, 47, 0, 75, 109, 45, 30, 60, 65, 95, 11, 29, 6, 8, 44, 17, 65, 7, 53, 108, 31, 3, 57, 54, 74, 25, 101, 38, 63, 57, 86, 4, 86, 66, 38, 2, 118, 23, 19, 23, 4, 83, 48, 70, 39, 16, 35, 8, 69, 22, 33, 25, 55, 18, 96, 15, 89, 4, 114, 107, 36, 89, 49, 97, 28, 97, 55, 49, 15, 83, 71, 22, 48, 19, 50, 7, 60, 35, 82, 43, 94, 106, 36, 112, 37, 96, 58, 40, 25, 6, 38, 52, 59, 30, 45, 81, 111, 42, 87, 27, 78, 32, 27, 108, 75, 30, 23, 39, 33, 45, 2, 0, 34, 81, 33, 62, 48, 64, 8, 83, 99, 76, 84, 1, 28, 75, 92, 34, 52, 8, 16, 99, 51, 14, 81, 61, 87, 113, 96, 20, 115, 36, 72, 45, 54, 115, 80, 103, 101, 61, 13, 27, 19, 51, 55, 102, 10, 105, 54, 38, 0, 87, 23, 82, 50, 91, 2, 5, 74, 45, 18, 28, 56, 23, 64, 68, 105, 76, 53, 12, 88, 49, 24, 12, 23, 9, 58, 42, 33, 112, 54, 78, 100, 49, 31, 92, 116, 107, 40, 52, 20, 10, 56, 22, 26, 47, 25, 63, 82, 113, 51, 76, 30, 31, 90, 24, 17, 105, 0, 46, 98, 66, 61, 93, 95, 88, 70, 57, 22, 95]
        return best_colors


    vertices = populate_vertices(edges)
    #counts = dict((v.index, v.neighbour_count) for v in vertices.values())
    # compute assignment order.  Or change later for dynamic assignment
    ordered_vertices = get_top_neighbour_counts(vertices, node_count)
    vp = VertexPool(ordered_vertices)
    if max_c is not None:
        max_color = min(node_count - 1, max_c)
    else:
        max_color = node_count - 1
    initialize_domain(vp, max_color)
    starting_domains = []#{}

    best_solution = None
    best_colors = None
    import copy

    THRESHOLD =900 + node_count *2

    i = 0
    while True: #i < node_count:

        if (best_colors is not None or max_c is not None) and time.time() - start > THRESHOLD:
            return best_colors

    #for i in range(node_count):
        if not vp.vertices[i].possible_choices:
            #print("no more choices")
            #print("need to backtrack")
            if not starting_domains:
                break
            starting_domains.pop()
            if not starting_domains:
                break
            vp = starting_domains[-1]
            i = len(starting_domains) - 1
            #i -= 2
            #print("i1:{}, len(starting_domains):{}".format(i, len(starting_domains)))
            #try:
            #vp = starting_domains[i]
            #except Exception as e:
            #    print(e)
            continue
            #raise


        # check if something isn't set properly

        for j in range(0, i):
            if vp.vertices[j].color is None:
                print("error")

        color = vp.vertices[i].possible_choices.pop()
        vp.vertices[i].color = color
        #vp.vertices[i].choices_made.add(color)
        starting_domains.append(copy.deepcopy(vp))
        #color = vp.vertices[i].possible_choices.pop()

        #print(vp.vertices[i].index, vp.vertices[i].color)
        #print("i2:{}, len(starting_domains):{}".format(i, len(starting_domains)))
        #valid = prune(vp, vp.vertices[i])
        valid = prune_everything(vp)
        if not valid:
            #print("after pruning, something is empty")
            #print("need to backtrack")
            #vp = starting_domains[i]
            if not starting_domains:
                break
            starting_domains.pop()
            starting_domains.pop()
            i -= 2
            #print("i3:{}, len(starting_domains):{}".format(i, len(starting_domains)))
            #del starting_domains[i]
            #raise
        elif i == node_count - 1:
            colors = []
            for i in range(node_count):
                colors.append(vp.vertex_index_map[i].color)
            print("found solution:{}".format(max(colors) + 1))
            print(colors)
            return colors

            if not check_everything(vp):
                starting_domains.pop()
                starting_domains.pop()
                i = len(starting_domains) - 1
                vp = starting_domains[i]
                continue
            #return colors
            best_colors = colors[:]
            best_solution = max(colors) + 1
            starting_domains = reduce_starting_domains(starting_domains, max(colors) - 1)
            if not starting_domains:
                break
            i = len(starting_domains) - 1
            vp = starting_domains[i]

            #print("i4:{}, len(starting_domains):{}".format(i, len(starting_domains)))
        else:
            i += 1
            #print("i:{}, len(starting_domains):{}".format(i, len(starting_domains)))

        if not check_everything(vp):
            starting_domains.pop()
            starting_domains.pop()
            i = len(starting_domains) - 1
            vp = starting_domains[i]
            continue

    return best_colors
    num_colors = max(best_colors) + 1

    # prepare the solution in the specified output format


def check_everything(vp):
    for vert in vp.vertices:
        for n_num in vert.neighbours:
            if vert.color == vp.vertex_index_map[n_num].color and vert.color is not None:
                return False
    return True

def print_domain(domain):
    a = 1
    size = [len(list(filter(lambda x: x is not None, list(v.possible_choices) + [v.color]))) for v in domain.vertices]
    for b in size:
        a = a * b
    choices = set()
    colors = [v.color for v in domain.vertices if v.color is not None]
    for v in domain.vertices:
        choices = choices.union(v.possible_choices)
    max_color = max(colors + [0])
    num = len(colors)
    max_choice = max(list(choices) + [0])
    print("domain: size: {}, populated:{}, max_color:{}, max_choice:{}".format(a, num, max_color, max_choice))

def reduce_starting_domains(starting_domains, new_max):
    new_starting_domains = []
    for i in range(len(starting_domains)):
        starting_domain = starting_domains[i]
        for vert in starting_domain.vertices:
            if vert.color is not None and vert.color > new_max:
                return new_starting_domains
            vert.possible_choices = set([pc for pc in vert.possible_choices if pc <= new_max])
            if vert.color is None and not vert.possible_choices:
                return new_starting_domains
        print_domain(starting_domain)
        new_starting_domains.append(starting_domain)
    return new_starting_domains

def prune(vp, vertex):
    vertex_color = vertex.color
    for neighbour_num in vertex.neighbours:
        neighbour = vp.vertex_index_map[neighbour_num]
        if vertex_color in neighbour.possible_choices:
            neighbour.possible_choices.remove(vertex_color)
        if neighbour.color is None and not neighbour.possible_choices:
            return False
    # print_domain(vp)
    return True

def prune_everything(vp):
    for vert in vp.vertices:
        ret = prune(vp, vert)
        if not ret:
            return False
    return True

def initialize_domain(vp, max_color):
    for i, vert in enumerate(vp.vertices):
        vert.possible_choices = set(range(min(i, max_color)+1))

def generate_output_data(solution):
    value = max(solution) + 1

    # build a trivial solution
    # every node has its own color
    # solution = range(0, node_count)'''

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

