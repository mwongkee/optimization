from collections import namedtuple, defaultdict
Node = namedtuple("Node", ['key', 'value', 'room', 'estimate'])


def breadth_first(item_count, capacity, items):
    # /10 -
    # /10 -
    # /10 -
    # /10 -
    # /10 -
    # /10 -

    items = list(reversed(sorted(items, key=lambda x: float(x.value) / float(x.weight))))

    key = ()

    nodes = defaultdict(list)# { level: [nodes_at_level]}
    head = Node(key, 0, capacity, None)
    max_node = [head]
    nodes[0] = [head]

    for level in range(item_count):
        branch_level(level, max_node, items, nodes)

    breadth_first_branch_from_node(head, max_node, items, nodes)

    value = max_node[0].value
    taken = [0] * item_count
    for i, val in enumerate(max_node[0].key):
        ind = items[i].index
        taken[ind] = val
    return value, taken


def branch_level(level, max_node, items, nodes):
    #print('branching level {}'.format(level))
    for node in nodes[level]:
        breadth_first_branch_from_node(node, max_node, items, nodes)
    nodes[level+1] = list(filter(lambda n: n.estimate >= max_node[0].value, nodes[level+1]))
    #print('num_nodes:{}/{}', len(nodes[level+1]), 2**(level+1))



def breadth_first_branch_from_node(prev_node, max_node, items, nodes):
    key1 = prev_node.key + (1,)
    key2 = prev_node.key + (0,)
    i = len(prev_node.key)
    item = items[i]
    node1 = Node(key1, prev_node.value + item.value, prev_node.room - item.weight, compute_estimate_fractional_quantities_sorted_by_density(items[i:], prev_node, True))
    node2 = Node(key2, prev_node.value, prev_node.room, compute_estimate_fractional_quantities_sorted_by_density(items[i:], prev_node, False))
    #print(node1)
    #print(node2)

    level = len(key1)

    if node1.room >= 0 and node1.value > max_node[0].value: # new max
        max_node[0] = node1
        #print(node1)

    if node1.estimate >= max_node[0].value and node1.room >= 0:
        nodes[level].append(node1)

    if node2.estimate >= max_node[0].value and node2.room >= 0:
        nodes[level].append(node2)

def depth_first(item_count, capacity, items):
    # 10/10 -
    # /10 - super slow
    # /10 -
    # /10 -
    # /10 -
    # /10 -


    key = ()

    nodes = {}
    head = Node(key, 0, capacity, None)
    max_node = [head]
    nodes[head.key] = head

    max_node = depth_first_branch_from_node(head, max_node, items, nodes)
    # print(max_node)

    value = max_node[0].value
    taken = [0] * item_count
    for i, val in enumerate(max_node[0].key):
        taken[i] = val
    return value, taken

def depth_first_branch_from_node(prev_node, max_node, items, nodes):
    key1 = prev_node.key + (1,)
    key2 = prev_node.key + (0,)
    i = len(prev_node.key)
    item = items[i]
    node1 = Node(key1, prev_node.value + item.value, prev_node.room - item.weight, compute_estimate_infinit_capacity(items[i:], prev_node))
    node2 = Node(key2, prev_node.value, prev_node.room, compute_estimate_infinit_capacity(items[i + 1:], prev_node))
    # print(node1)
    # print(node2)

    # nodes[key1] = node1
    # nodes[key2] = node2

    if node1.room >= 0 and node1.value > max_node[0].value: # new max
        max_node[0] = node1
        #print(node1)

    if node1.room > 0 and node1.estimate >= max_node[0].value and i < len(items)-1:
        depth_first_branch_from_node(node1, max_node, items, nodes)

    if node2.room > 0 and node2.estimate >= max_node[0].value and i < len(items)-1:
        depth_first_branch_from_node(node2, max_node, items, nodes)
    return max_node

def compute_estimate_infinit_capacity(remaining_items, prev_node):
    remaining_sum = sum([a.value for a in remaining_items])
    return prev_node.value + remaining_sum

def compute_estimate_fractional_quantities_sorted_by_density(remaining_items, prev_node, take_item):
    room = prev_node.room
    value = prev_node.value
    if take_item:
        room -= remaining_items[0].weight
        value += remaining_items[0].value

    #remaining_items = reversed(sorted(remaining_items[1:], key=lambda x: float(x.value)/float(x.weight)))

    for item in remaining_items:
        if item.weight <= room:
            room -= item.weight
            value += item.value
        else:
            fraction = float(room) / float(item.weight)
            value += fraction * item.value
            return value
    return value