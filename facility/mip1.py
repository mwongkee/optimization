#    customers c(j)  which warehouse boolean
#    distance  d(i,j)
#
#   minimize sum(   D(i,j) * c(j)
#
#
#
#
#
#
#
#
def mip1(facilities, customers):
    return []


from collections import namedtuple
import math
from cvxpy import *
import numpy as np
import random

def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def weight_length(num_facilities, num_customers, customer, facility):
    return length(customer.location, facility.location)

def weight_length_plus_some_setup(num_facilities, num_customers, customer, facility):
    dist = length(customer.location, facility.location)
    setup = facility.setup_cost
    factor = 2
    return dist + setup/factor

def weight_random(num_facilities, num_customers, customer, facility):
    return random.random()


def compute_neighbours(customers, facilities, neighbour_func, num_neighbours):
    neighbours = {}
    num_customers = len(customers)
    num_facilities = len(facilities)
    for c_i, c in enumerate(customers):
        this_neighbours = []
        for f_i, f in enumerate(facilities):
            this_neighbours.append(neighbour_func(num_facilities, num_customers, c, f))
        sorted_ind = np.array(this_neighbours).argsort()
        neighbours[c_i] = set(sorted_ind[:num_neighbours])
    return neighbours



def run1(facilities, customers, neighbour_func, num_neighbours):
    Point = namedtuple("Point", ['x', 'y'])
    Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
    Customer = namedtuple("Customer", ['index', 'demand', 'location'])

    neighbours = compute_neighbours(customers, facilities, neighbour_func, num_neighbours)

    M = len(customers) + 1
    # variables
    from collections import defaultdict
    in_facility = defaultdict(dict)
    for f_i, f in enumerate(facilities):
        for c_i, c in enumerate(customers):
            if f_i in neighbours.get(c_i):
                in_facility[f_i][c_i] = Bool()

    in_business = [Bool() for f in facilities]

    # under capacity
    constraints = []

    for f_i, f in enumerate(facilities):
        usage = None
        fac_bus = None
        for c_i, c in enumerate(customers):
            if c_i in in_facility[f_i]:
                if usage is None:
                    usage = in_facility[f_i][c_i] * c.demand
                else:
                    usage += in_facility[f_i][c_i] * c.demand
                if fac_bus is None:
                    fac_bus = in_facility[f_i][c_i]
                else:
                    fac_bus += in_facility[f_i][c_i]

        if usage is not None:
            constraint = usage <= f.capacity
            constraints.append(constraint)

            # only in business can serve
            constraint = fac_bus <= in_business[f_i] * M
            constraints.append(constraint)
        else:
            constraint = in_business[f_i] == 0
            constraints.append(constraint)

    # only served by one
    for c_i, c in enumerate(customers):
        served_by = None
        for f_i, f in enumerate(facilities):
            if c_i in in_facility[f_i]:
                if served_by is None:
                    served_by = in_facility[f_i][c_i]
                else:
                    served_by += in_facility[f_i][c_i]

        constraint = served_by == 1
        constraints.append(constraint)

    cost = None
    for f_i, f in enumerate(facilities):
        for c_i, c in enumerate(customers):
            distance = length(f.location, c.location)
            if c_i in in_facility[f_i]:
                if cost is None:
                    cost = in_facility[f_i][c_i] * distance
                else:
                    cost += in_facility[f_i][c_i] * distance


    for f_i, f in enumerate(facilities):
        cost += in_business[f_i] * facilities[f_i].setup_cost

    for con in constraints:
        print(con.__str__())

    print(cost.__str__())

    # Form objective.
    obj = Minimize(cost)

    # Form and solve problem.
    prob = Problem(obj, constraints)
    print(prob.solve())
    print(in_facility)
    print(in_business)

    customer_facilities = []

    customer_to_facility = {}
    for f_i, customers_dict in in_facility.items():
        for c_i, var in customers_dict.items():
            try:
                if var.value > 0.5:
                    customer_to_facility[c_i] = f_i
            except Exception as e:
                print(e)

    for c_i, c in enumerate(customers):
        #customer_facilities.append([in_facility[j][i].value > 0.5 for j in range(len(facilities))].index(True))
        customer_facilities.append(customer_to_facility.get(c_i))
    return customer_facilities


if __name__ == '__main__':
    run1()

