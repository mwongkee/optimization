#from drawer import draw_solution
from collections import defaultdict
import numpy as np
import random

# 1 - 462.104188426 7/10 need 430    3opt 446 sa 428
# 2 - 22059.0933893 7/10 need 20800 3opt sa 20775
# 3 - 32131.511588 7/10  need 30000  3opt 31698 sa
# 4 - 39898.070283 7/10 need 37600
# 5 - 355298.600001  7/10 need 323000
# 6 -

def calc_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calc_total_distance(node_count, points, solution):
    obj = calc_distance(points[solution[-1]], points[solution[0]])
    for index in range(0, node_count-1):
        obj += calc_distance(points[solution[index]], points[solution[index+1]])
    return obj


def get_neighbours2(points, n=1):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    max_x = max(xs)
    min_x = min(xs)
    max_y = max(ys)
    min_y = min(ys)
    buckets = defaultdict(dict)

    for i in range(n):
        for j in range(n):
            buckets[i][j] = set()

    for i, p in enumerate(points):
        x_bucket = int(np.floor(n * abs(p.x - 0.0000001 - min_x) / (max_x - min_x)))
        y_bucket = int(np.floor(n * abs(p.y - 0.0000001 - min_y) / (max_y - min_y)))

        buckets[x_bucket][y_bucket].add(p)

    neighbours = defaultdict(dict)
    num_points = len(points)
    k = 0
    for i in range(n):
        for j in range(n):

            to_diff = set()
            to_diff = to_diff.union(buckets[i][j])
            if i > 0:
                to_diff = to_diff.union(buckets[i-1][j])
                if j > 0:
                    to_diff = to_diff.union(buckets[i - 1][j-1])
                if j < n-1:
                    to_diff = to_diff.union(buckets[i - 1][j + 1])
            if i < n-1:
                to_diff = to_diff.union(buckets[i + 1][j])
                if j > 0:
                    to_diff = to_diff.union(buckets[i + 1][j-1])
                if j < n-1:
                    to_diff = to_diff.union(buckets[i + 1][j + 1])
            if j > 0:
                to_diff = to_diff.union(buckets[i][j - 1])
            if j < n-1:
                to_diff = to_diff.union(buckets[i][j + 1])

            for p1 in buckets[i][j]:
                k += 1
                #print('{}/{}'.format(k, num_points))
                for p2 in to_diff:
                    if p1.index in neighbours[p2.index]:
                        neighbours[p1.index][p2.index] = neighbours[p2.index][p1.index]
                    elif p1.index != p2.index:
                        dist = calc_distance(p1, p2)
                        neighbours[p1.index][p2.index] = dist
    return neighbours



def generate_initial_solution(node_count, points, neighbours):
    to_place = set(range(node_count))

    solution = [to_place.pop()]
    while to_place:
        #if np.mod(len(to_place),1000) == 0 or len(to_place) < 1000:
            #print(len(to_place))

        next = to_place.pop()
        #if len(to_place) < 5:
            #print(next)
        added = False
        if len(solution) == 1:
            if next in neighbours[solution[0]]:
                solution.append(next)
                added = True
        else:
            for i in range(len(solution) - 1):

                #if len(to_place) < 5:
                    #if next in neighbours[solution[i]]:
                    #    print(solution[i], neighbours[solution[i]])
                    #if next in neighbours[solution[i+1]]:
                    #    print(solution[i+1],neighbours[solution[i+1]] )


                if next in neighbours[solution[i]] and next in neighbours[solution[i+1]]:


                    solution.insert(i+1, next)
                    #print(solution)
                    added = True
                    break
        if not added:
            to_place.add(next)

    # verify
    for i in range(len(solution)-1):
        if solution[i+1] not in neighbours[solution[i]]:
            print('error')
    #print('initial solution')
    #print(solution)
    return solution

def simulated_annealing_solver(node_count, points, alpha, n=1, T_start=10, T_end=0.00001, initial=None, draw=False):
    #print('alpha:{} n={} T_start={} T_end={}'.format(alpha,n, T_start, T_end))
    #print('total iterations = {}'.format(np.log(T_end/T_start) / np.log (alpha)))
    neighbours = get_neighbours2(points)
    #print(neighbours)
    if initial is None:
        solution = generate_initial_solution(node_count, points, neighbours)
    else:
        solution = initial
    #solution = np.random.permutation(range(0, node_count)).tolist()
    #solution = [0,3,2,1,4]
    obj = best_obj = calc_total_distance(node_count, points, solution)
    if draw:
        fig = draw_solution(points, solution, 0, '', obj)
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(r'C:\working\tsp_{}.pdf'.format(node_count))
        pdf.savefig(fig)

    best_solution = None
    best_distance = None
    T = T_start
    iteration = 0
    while T > T_end:

        random_point = random.randint(0, node_count - 1)

        # rotate solution so that random point is at 0
        solution = solution[random_point:] + solution[:random_point]

        #print('random_point:{}'.format(random_point))
        #random_point = 0
        a = solution[0]
        #print(points[a])

        b = solution[1]
        neighs = neighbours[b]

        # remove 1 edges
        #for ind, dist in neighs.items():
        #    print(ind, dist)
        #    attempt_obj = obj - neighs[random_index][next_index]
        #    attempt_obj += neighs[random_index][ind]
        #    attempt_obj -=

        e1 = neighbours[a][b]

        neighbours_intersect = set(neighbours[a].keys()).intersection(set(neighbours[b].keys()))

        best_improvement = 0
        best_imp_i = None
        best_improvement_is = {}

        random_c = random.randint(0, len(neighbours_intersect) - 1)
        c = list(neighbours_intersect)[random_c]

        c_i = solution.index(c)
        if c_i == node_count - 1:
            continue
        d_i = c_i + 1
        d = solution[d_i]
        if d not in neighbours_intersect:
            continue

        neighbours_intersect_cd = neighbours_intersect.intersection(
            set(neighbours[c].keys()).intersection(set(neighbours[d].keys())))

        random_e = random.randint(0, len(neighbours_intersect_cd)-1)

        e = list(neighbours_intersect_cd)[random_e]
        e_i = solution.index(e)
        if e_i == node_count - 1:
            continue
        if e_i == c_i or e_i == d_i:
            continue
        f_i = e_i + 1
        f = solution[f_i]
        if f not in neighbours_intersect_cd:
            continue
        if f_i == c_i:
            continue

        if e_i < c_i:
            c_i, d_i, e_i, f_i = e_i, f_i, c_i, d_i
            c, d, e, f = e, f, c, d



            n = neighbours

            ab = n[a][b]
            cd = n[c][d]
            ef = n[e][f]

            imp = []
            # https://stackoverflow.com/questions/21205261/3-opt-local-search-for-tsp
            # 1    ab ce df  2opt
            imp.append(-n[c][e] - n[d][f] + cd + ef)
            # 2    ac bd ef  2opt
            imp.append(-n[a][c] - n[b][d] + ab + cd)
            # 3    ac be df
            imp.append(-n[a][d] - n[b][e] - n[c][f] + ab + cd + ef)
            # 4    ad eb cf
            imp.append(-n[a][d] - n[e][b] - n[c][f] + ab - cd + ef)
            # 5    ad ec bf
            imp.append(-n[a][d] - n[e][c] - n[b][f] + ab - cd + ef)
            # 6    ae db cf
            imp.append(-n[a][e] - n[d][b] - n[c][f] + ab - cd + ef)
            # 7    ae dc bf
            imp.append(-n[a][e] - n[d][c] - n[b][f] + ab + cd + ef)

            best_imp = max(imp)
            best_imp_i = imp.index(best_imp)

            # compute chance of taking it

            T = T * alpha
            prob = np.exp(best_imp/T)
            rand = random.random()
            if prob < rand:
                continue # not accepting

            a_i = 0
            b_i = 1
            s = solution
            if best_imp_i == 0:
                solution = s[:a_i+1] + s[b_i:c_i+1] + s[e_i:d_i-1:-1] + s[f_i:]
            elif best_imp_i == 1:
                solution = s[:a_i+1] + s[c_i:b_i-1:-1] + s[d_i:e_i+1] + s[f_i:]
            elif best_imp_i == 2:
                solution = s[:a_i+1] + s[c_i:b_i-1:-1] + s[e_i:d_i-1:-1] + s[f_i:]
            elif best_imp_i == 3:
                solution = s[:a_i+1] + s[d_i:e_i+1] + s[b_i:c_i+1] + s[f_i:]
            elif best_imp_i == 4:
                solution = s[:a_i+1] + s[d_i:e_i+1] + s[c_i:b_i-1:-1] + s[f_i:]
            elif best_imp_i == 5:
                solution = s[:a_i+1] + s[e_i:d_i-1:-1] + s[b_i:c_i+1] + s[f_i:]
            elif best_imp_i == 6:
                solution = s[:a_i+1] + s[e_i:d_i-1:-1] + s[c_i:b_i-1:-1] + s[f_i:]

            for i in range(len(solution)-1):
                if solution[i] not in neighbours[solution[i+1]]:
                    raise
                if solution[i+1] not in neighbours[solution[i]]:
                    raise

            #draw_solution(points, solution, a, b, c, d)
            if len(solution) != node_count:
                raise

            for i in range(node_count):
                if solution.index(i) == -1:
                    raise


            #else:
                #raise
            #    continue
            #if np.mod(iter, 100) == 0:
            #    draw_solution(points, solution, a, b, c, d)
            dist = calc_total_distance(node_count, points, solution)
            #print(iteration, dist)
            #print(solution)
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_solution = solution[:]

            if draw:
                fig = draw_solution(points, solution, iteration, T, dist)
                pdf.savefig(fig)
        iteration += 1

    calc_total_distance(node_count, points, solution)
    if draw:
        pdf.close()
    return best_solution

        # 0->13-49->14-50




