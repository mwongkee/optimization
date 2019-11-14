from collections import namedtuple
from simulated_annealing_tsp import simulated_annealing_solver as sa
from simulated_annealing_tsp import calc_total_distance

def initialize(customer_count, vehicle_count, vehicle_capacity, remaining_customers):
    vehicle_tours = []
    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            import random
            #order = sorted(remaining_customers, key=lambda customer: -customer.demand)
            order = list(remaining_customers)
            random.shuffle(list(remaining_customers))#, key=lambda customer: -customer.demand)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used

            # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == customer_count - 1
    return vehicle_tours


Point = namedtuple("Point", ['x', 'y', 'index'])


def optimize_tsps(vehicle_tours, depot):
    updated_vehicle_tours = []
    for tour in vehicle_tours:
        tour1 = tour[:] + [depot]
        index_to_cust = dict((c.index, c) for c in tour1)
        node_count = len(tour1)
        index_mapping = {}
        points = []
        for i, cust in enumerate(tour1):
            points.append(Point(cust.x, cust.y, i))
            index_mapping[i] = cust.index
        alpha = 0.99
        if node_count < 8:
            indices = brute_force_tsp(node_count, points)
            tour1 = [index_to_cust[index_mapping[i.index]] for i in indices]
        else:
            indices = sa(node_count, points, alpha, n=1, T_start=10, T_end=0.01, initial=None, draw=False)
            tour1 = [index_to_cust[index_mapping[i]] for i in indices]

        depot_index = [i for i, c in enumerate(tour1) if c.index == 0][0]

        updated_vehicle_tours.append(tour1[depot_index + 1:] + tour1[:depot_index])
        #print(tour)
    return updated_vehicle_tours

import math

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


def calculate_obj(vehicle_count, vehicle_tours, depot):
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(depot,vehicle_tour[0])
            for i in range(0, len(vehicle_tour)-1):
                obj += length(vehicle_tour[i],vehicle_tour[i+1])
            obj += length(vehicle_tour[-1],depot)
    return obj

import numpy as np
def simulated_annealing_solver(node_count, points, alpha, n=1, T_start=10, T_end=0.00001, initial=None, draw=False):
    print('alpha:{} n={} T_start={} T_end={}'.format(alpha,n, T_start, T_end))
    print('total iterations = {}'.format(np.log(T_end/T_start) / np.log (alpha)))

def run1(customer_count, vehicle_count, vehicle_capacity, depot, remaining_customers):
    vehicle_tours = initialize(customer_count, vehicle_count, vehicle_capacity, remaining_customers)
    best_obj = calculate_obj(vehicle_count, vehicle_tours, depot)
    best_vehicle_tours = vehicle_tours[:]

    alpha = 0.999
    T_start = 100000
    T_end = 10

    import random
    T = T_start
    iteration = 0
    while T > T_end:
        #print(T)
        #random_customer_i = random.randint(1, vehicle_count )
        vehicle_tours_copy = vehicle_tours[:]
        print(T)
        T = T * alpha
        num_cust_swap = 2
        random_customer_is = list(set(random.sample(range(1, customer_count), num_cust_swap)))

        random_vehicle_is = [random.randint(0, vehicle_count - 1) for _ in range(num_cust_swap)]

        for swap_num in range(num_cust_swap):
            for tour_ind, tour in enumerate(vehicle_tours_copy):
                cust_ind_list = [i for i, c in enumerate(tour) if c.index == random_customer_is[swap_num]]
                if cust_ind_list:
                    random_customer = tour[cust_ind_list[0]]
                    vehicle_tours_copy[tour_ind] = tour[:cust_ind_list[0]] + tour[cust_ind_list[0]+1:]
                    vehicle_tours_copy[random_vehicle_is[swap_num]] = vehicle_tours_copy[random_vehicle_is[swap_num]] + [random_customer]
                    break

        # check capacities
        exceeded = False
        for tour in vehicle_tours_copy:
            if sum([c.demand for c in tour]) > vehicle_capacity:
                exceeded = True
                break
        if exceeded:
            continue


        updated_vehicle_tours = optimize_tsps(vehicle_tours_copy, depot)
        obj = calculate_obj(vehicle_count, updated_vehicle_tours, depot)


        prob = np.exp((best_obj - obj)/ T)
        rand = random.random()
        if prob < rand:
            continue  # not accepting

        print(obj)
        if obj < best_obj:
            best_obj = obj
            best_vehicle_tours = updated_vehicle_tours[:]

        vehicle_tours = updated_vehicle_tours[:]

    return best_vehicle_tours





def brute_force_tsp(node_count, points):
    #print(points)
    #print(points)
    from itertools import permutations
    perms = list(permutations(range(1, len(points))))

    best_solution = None
    best_obj = None

    for perm in perms:
        solution = [0] + list(perm)
        obj = calc_total_distance(node_count, points, solution)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_solution = solution[:]

    return [points[i] for i in best_solution]


