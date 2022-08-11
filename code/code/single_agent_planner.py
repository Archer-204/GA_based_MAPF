import heapq
import numpy as np
import random
import math


def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
                    or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3: Return a table that contains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    constraint_table = dict()
    for cur_constraint in constraints:
        if cur_constraint['agent'] == agent:
            # print('(time, loc):', [cur_constraint['timestep'], cur_constraint['loc']])
            # print('constraint', cur_constraint)
            if cur_constraint['timestep'] in constraint_table:
                constraint_table[cur_constraint['timestep']][tuple(cur_constraint['loc'])] = cur_constraint['positive']
            else:
                temp = dict()
                temp[tuple(cur_constraint['loc'])] = cur_constraint['positive']
                constraint_table[cur_constraint['timestep']] = temp
    return constraint_table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    print("ThePath: ")
    print(path)
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.
    # print('time = ', next_time)
    # print(['inside is_constrained', curr_loc, next_loc, next_time])
    if next_time in constraint_table:
        # print('loc = ', tuple([curr_loc, next_loc]))
        # print('constraint_table[next_time] = ', constraint_table[next_time])
        if tuple([next_loc]) in constraint_table[next_time]:
            if not constraint_table[next_time][tuple([next_loc])]:
                return True
        elif tuple((curr_loc, next_loc)) in constraint_table[next_time]:
            if not constraint_table[next_time][tuple((curr_loc, next_loc))]:
                return True
    return False


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

class Genome:
    # TODO: changed here, need test
    def __init__(self, len, start, constraint_table):
        self.bits = []
        self.fitness = 0
        self.stop = 0
        timestep = 0
        current_loc = start
        for g in range((int)(len / 2)):
            timestep += 1
            x = random.randint(0, 1)
            y = random.randint(0, 1)
            next_loc = get_GA_location(current_loc, [x, y])
            while is_constrained(current_loc, next_loc, timestep, constraint_table):
                x = random.randint(0, 1)
                y = random.randint(0, 1)
                next_loc = get_GA_location(current_loc, [x, y])
            self.bits.append(x)
            self.bits.append(y)
            current_loc = next_loc


# TODO: need test
def get_GA_location(current_loc, move):
    position = current_loc
    if move[0] == 0 and move[1] == 0:
        position[1] += 1
    elif move[0] == 0 and move[1] == 1:
        position[0] -= 1
    elif move[0] == 1 and move[1] == 0:
        position[1] -= 1
    else:
        position[0] += 1
    return position


# TODO: need test
def get_GA_loc_at_time(bits, start, time):
    cur_time = 0
    cur_loc = start
    s = 0
    while cur_time < time:
        cur_time += 1
        move = [bits[s], bits[s + 1]]
        cur_loc = get_GA_location(cur_loc, move)
        s += 2
    return cur_loc


def build_genList(num, start, constraint_table):
    genList = []
    for i in range(2 * num):
        gen = Genome(num, start, constraint_table)
        genList.append(gen)
    return genList


def detect(map, gen, start, end):
    position = []
    position.append(start[0])
    position.append(start[1])
    print(start)
    print(map)
    for j in range(int(len(gen.bits) / 2)):
        g = [gen.bits[j * 2], gen.bits[j * 2 + 1]]
        if g[0] == 0 and g[1] == 0:
            position[1] += 1
        elif g[0] == 0 and g[1] == 1:
            position[0] -= 1
        elif g[0] == 1 and g[1] == 0:
            position[1] -= 1
        else:
            position[0] += 1

        # if ended with a obstacle
        if map[position[0]][position[1]]:
            gen.fitness = 1 / (abs(end[0] - position[0]) + abs(end[1] - position[1]) + 1)
            gen.stop = j
            break
        elif (position[0], position[1]) == end:
            gen.fitness = 1
            gen.stop = j
            print("reach the goal: stop is ")
            print(gen.stop)
            return 1
        else:
            print("safe to go: ")
            print((position[0], position[1]))
    # if gen.stop == 0:
    #     gen.fitness = 1 / (abs(end[0] - position[0]) + abs(end[1] - position[1]) + 1)
    #     gen.stop = int(len(gen.bits) / 2)
    return 0


def select(genList):
    total_fit = 0
    for g in genList:
        total_fit += g.fitness
    piece = random.random() * total_fit
    temp = 0
    for g in genList:
        temp = temp + g.fitness
        if temp >= piece:
            return g


# One-point crossover
def crossover(dad, mum, rate):
    child1 = []
    child2 = []
    rn = random.random()
    if rn < rate:
        pt = random.randint(0, len(dad.bits))
        for i in range(pt):
            child1.append(dad.bits[i])
            child2.append(mum.bits[i])
        for i in range(pt, len(dad.bits)):
            child1.append(mum.bits[i])
            child2.append(dad.bits[i])
    else:
        child1 = dad.bits
        child2 = mum.bits
    return child1, child2


def mutate(gen, rate):
    for i in range(0, len(gen.bits)):
        if random.random() < rate:
            if gen.bits[i] == 1:
                gen.bits[i] = 0
            else:
                gen.bits[i] = 1


# TODO: need test
def fix_gene_with_constraint(bits, start, constraint_table):
    timestep = 0
    new_bits = []
    current_loc = start
    s = 0
    for g in range((int)(len(bits) / 2)):
        timestep += 1
        x = bits[s]
        y = bits[s + 1]
        next_loc = get_GA_location(current_loc, [x, y])
        while is_constrained(current_loc, next_loc, timestep, constraint_table):
            x = random.randint(0, 1)
            y = random.randint(0, 1)
            next_loc = get_GA_location(current_loc, [x, y])
        new_bits.append(x)
        new_bits.append(y)
        current_loc = next_loc
        s += 2
    return new_bits


def generatePath(map, bits, start, end):
    path = []
    path.append(start)
    s = 0
    cur = 0

    while (path[cur][0], path[cur][1]) != end:
        if [bits[s], bits[s + 1]] == [0, 0]:
            move = (path[cur][0], path[cur][1] + 1)
            path.append(move)
        elif [bits[s], bits[s + 1]] == [0, 1]:
            move = (path[cur][0] - 1, path[cur][1])
            path.append(move)
        elif [bits[s], bits[s + 1]] == [1, 0]:
            move = (path[cur][0], path[cur][1] - 1)
            path.append(move)
        elif [bits[s], bits[s + 1]] == [1, 1]:
            move = (path[cur][0] + 1, path[cur][1])
            path.append(move)
        s += 2
        cur += 1
    print("path")
    print(path)
    return path


def a_star(map, start, end, h_values, agent, constraints):
    """ map      - binary obstacle map
        start   - start position(tuple)
        goal_loc    - goal position(tuple)
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep(dictionary)
    """

    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.
    # print("Start Location:")
    # print(start_loc[0])
    # print(start_loc[1])
    print("1")
    print(start)
    # num of chromosomes in a gene
    start = [start[0], start[1]]
    end = [end[0], end[1]]
    gn = len(map) * len(map[0]) * 0.5
    gn = int(gn)
    # build constraint table for current agent
    constraint_table = build_constraint_table(constraints, agent)
    print("constraint_table", constraint_table)

    genList = build_genList(gn, start, constraint_table)
    flag = 0
    for g in genList:
        flag = detect(map, g, start, end)
        if flag == 1:
            print("Find It!")
            print(g.stop)
            print(g.bits)
            return generatePath(map, g.bits, start, end)

    newlist = []
    time = 0
    while flag == 0:
        time += 1
        child_n = 0
        while child_n < gn * 2:
            dad = select(genList)
            mum = select(genList)

            child1 = Genome(gn, start, constraint_table)
            child2 = Genome(gn, start, constraint_table)
            print("2")
            print(start)
            child1.bits, child2.bits = crossover(dad, mum, 0.5)
            detect(map, child1, start, end)
            detect(map, child2, start, end)

            mutate(child1, 0.3)
            child1.bits = fix_gene_with_constraint(child1.bits, start, constraint_table)

            mutate(child2, 0.3)
            child2.bits = fix_gene_with_constraint(child2.bits, start, constraint_table)
            print("3")
            print(start)
            flag = detect(map, child1, start, end)
            print("4")
            print(start)
            if flag == 1:
                print("Find It!")
                print(f'number of iteration is {time}')
                print(f'effective chromosome length {child1.stop + 1}')
                return generatePath(map, child1.bits, start, end)
            print("4")
            print(start)
            flag = detect(map, child2, start, end)
            if flag == 1:
                print("Find It!")
                print(f'number of iteration is {time}')
                print(f'effective chromosome length {child2.stop + 1}')
                return generatePath(map, child2.bits, start, end)

            newlist.append(child1)
            newlist.append(child2)

            child_n += 2
        genList = newlist
        newlist = []

# def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
#     """ my_map      - binary obstacle map
#         start_loc   - start position
#         goal_loc    - goal position
#         agent       - the agent that is being re-planned
#         constraints - constraints defining where robot should or cannot go at each timestep
#     """
#
#     ##############################
#     # Task 1.1: Extend the A* search to search in the space-time domain
#     #           rather than space domain, only.
#
#     open_list = []
#     closed_list = dict()
#     earliest_goal_timestep = 0
#     h_value = h_values[start_loc]
#     constraint_table = build_constraint_table(constraints, agent)
#     # print("constraint_table", constraint_table)
#     root = {'loc': start_loc,
#             'g_val': 0,
#             'h_val': h_value,
#             'parent': None,
#             'time_step': 0}
#     push_node(open_list, root)
#     closed_list[(root['loc'], root['time_step'])] = root
#     if len(constraint_table) > 0:
#         sorted(constraint_table.keys())
#         earliest_goal_timestep = list(constraint_table.keys())[-1]
#     while len(open_list) > 0:
#         curr = pop_node(open_list)
#         #############################
#         # Task 1.4: Adjust the goal test condition to handle goal constraints
#         # if curr['loc'] == goal_loc and curr['time_step'] >= 16:
#         #     return get_path(curr)
#         # if curr['loc'] == goal_loc:
#         #     return get_path(curr)
#         if curr['loc'] == goal_loc and curr['time_step'] >= earliest_goal_timestep:
#             return get_path(curr)
#         elif curr['time_step'] >= len(my_map) * len(my_map[0]):
#             break
#
#         for dir in range(5):
#             if dir < 4:
#                 child_loc = move(curr['loc'], dir)
#             else:
#                 child_loc = curr['loc']
#
#             if my_map[child_loc[0]][child_loc[1]]:
#                 continue
#             child = {'loc': child_loc,
#                      'g_val': curr['g_val'] + 1,
#                      'h_val': h_values[child_loc],
#                      'parent': curr,
#                      'time_step': curr['time_step'] + 1}
#
#             if not is_constrained(curr['loc'], child['loc'], child['time_step'], constraint_table):
#                 # print([agent, curr['loc'], child['loc'], child['time_step']])
#                 if (child['loc'], child['time_step']) in closed_list:
#                     # print('open list', open_list)
#                     existing_node = closed_list[(child['loc'], child['time_step'])]
#                     if compare_nodes(child, existing_node):
#                         closed_list[(child['loc'], child['time_step'])] = child
#                         push_node(open_list, child)
#                 else:
#                     # print('closed list', closed_list)
#                     closed_list[(child['loc'], child['time_step'])] = child
#                     push_node(open_list, child)
#
#     return None  # Failed to find solutions
