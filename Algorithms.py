import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict


class Node():
    def __init__(self, state=None, parent=None, action=-1, cost=0, g_val=0, h_val=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.g_value = g_val
        self.h_value = h_val


class WeightedAStarAgent():

    def __init__(self):
        self.env = None
        self.env: CampusEnv

    @staticmethod
    def manhattan_distance(point1, point2):  # point = (X,Y)
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])  # point[0] = X , point[1] = Y

    @staticmethod
    def solution(node: Node, expanded) -> Tuple[list[int], int, int]:
        total_cost = 0
        actions = []

        while node.parent is not None:
            actions.append(node.action)
            total_cost += node.cost
            node = node.parent
        

        actions.reverse()
        return actions, total_cost, expanded

    def h_campus(self, state):
        min_manhattan = np.inf
        state_point = self.env.to_row_col(state)
        for goal_state in self.env.get_goal_states():
            goal_point = self.env.to_row_col(goal_state)
            curr_distance = self.manhattan_distance(state_point, goal_point)
            if curr_distance < min_manhattan:
                min_manhattan = curr_distance
        return min(100, min_manhattan)

    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        start_state = self.env.get_initial_state()
        start_node = Node(state=start_state, h_val=self.h_campus(start_state))

        expanded = 0
        open_list = heapdict.heapdict()
        open_list[start_node] = (start_node.h_value * h_weight , start_node.state)
        closed_list = {}
        
        while open_list:
            current_node, (current_f_value, _) = open_list.popitem()
            closed_list[current_node.state] = current_f_value
            
            if self.env.is_final_state(current_node.state):
                return self.solution(current_node, expanded)

            expanded += 1
            for action in self.env.succ(current_node.state):
                self.env.reset()
                self.env.set_state(current_node.state)
                new_state, cost, terminated = self.env.step(action)

                if new_state == current_node.state or (terminated and not self.env.is_final_state(new_state)): 
                    continue

                new_h = self.h_campus(new_state)
                new_g = current_node.g_value + cost
                new_f = (new_h * h_weight) + (new_g * (1 - h_weight))
                
                in_open_list = any(node.state == new_state for node in open_list)
                in_closed_list = new_state in closed_list
                new_node = Node(new_state, current_node, action, cost, new_g, new_h)

                if not in_closed_list and not in_open_list:
                    open_list[new_node] = (new_f, new_state)
                    
                elif in_open_list:
                    for node in list(open_list.keys()):
                        if new_state == node.state and new_f < open_list[node][0]:
                            open_list[new_node] = (new_f, new_state)
                            del open_list[node]
                            break

                elif in_closed_list:
                    if new_f < closed_list[new_state]:
                        open_list[new_node] = (new_f, new_state)
                        del closed_list[new_state]

        return [], -1, -1
    