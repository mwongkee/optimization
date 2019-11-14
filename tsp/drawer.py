import matplotlib.pyplot as plt

def draw_solution(points, solution, iter, T, dist):#, a, b, c, d):
    fig =  plt.figure()
    x = [points[i].x for i in solution]
    y = [points[i].y for i in solution]

    plt.scatter(x, y)

    for i in range(len(solution) - 1):
        p1 = points[solution[i]]
        p2 = points[solution[i+1]]
        plt.plot([p1.x, p2.x], [p1.y, p2.y], 'k-', lw=2)

    p1 = points[solution[-1]]
    p2 = points[solution[0]]
    plt.plot([p1.x, p2.x], [p1.y, p2.y], 'k-', lw=2)
    plt.title('Iteration:{} T:{} Dist:{}'.format(iter, T, dist))
    return fig
    #plt.scatter([points[a].x, points[b].x, points[c].x, points[d].x],
    #            [points[a].y, points[b].y, points[c].y, points[d].y], marker='X')

    #plt.show()