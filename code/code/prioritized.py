import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        constraints = []
        max_time = len(self.my_map[0]) * len(self.my_map)
        # ---------------- Test constraints table --------------------
        # constraint1 = {'agent': 0,
        #                'loc': [(1, 5)],
        #                'timestep': 4,
        #                'positive': False}
        # constraints.append(constraint1)

        # constraint2 = {'agent': 1,
        #                'loc': [(1, 2), (1, 3)],
        #                'timestep': 1,
        #                'positive': True}
        # constraints.append(constraint2)

        # constraint3 = {'agent': 0,
        #                'loc': [(1, 5)],
        #                'timestep': 10}
        # constraints.append(constraint3)

        # constraint4 = {'agent': 1,
        #                'loc': [(1, 2)],
        #                'timestep': 2}
        # constraints.append(constraint4)
        #
        # constraint5 = {'agent': 1,
        #                'loc': [(1, 3)],
        #                'timestep': 2}
        # constraints.append(constraint5)
        #
        # constraint6 = {'agent': 1,
        #                'loc': [(1, 4)],
        #                'timestep': 2}
        # constraints.append(constraint6)

        for i in range(self.num_of_agents):  # Find path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)
            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            ##############################
            # Task 2: Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches

            for move_time in range(1, len(path)):
                for j in range(i + 1, self.num_of_agents):
                    my_constraint1 = {'agent': j,
                                      'loc': [path[move_time]],
                                      'timestep': move_time,
                                      'positive': False}
                    constraints.append(my_constraint1)

                    my_constraint2 = {'agent': j,
                                      'loc': [path[move_time], path[move_time - 1]],
                                      'timestep': move_time,
                                      'positive': False}
                    constraints.append(my_constraint2)

                    for future_time in range(len(path)-1, max_time):
                        my_constraint3 = {'agent': j,
                                          'loc': [path[-1]],
                                          'timestep': future_time,
                                          'positive': False}
                        constraints.append(my_constraint3)

            ##############################

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
