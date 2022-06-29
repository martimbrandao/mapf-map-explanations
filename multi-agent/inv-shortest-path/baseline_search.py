import yaml
import networkx as nx
import cvxpy as cp
import numpy as np
import argparse
import subprocess
import resource
from path import *
from additional import visualize
import explanations_multi
import copy
import time
import pdb

MAX_TIME = 60 * 5     # 5 minutes

def main_inv_mapf(problem_file, verbose=False, animate=False, question_partial_plan=False):
    problem_fullpath = EXAMPLES_PATH + "/" + problem_file
    return main_inv_mapf_fullpath(problem_fullpath, verbose, animate, question_partial_plan)


def main_inv_mapf_fullpath(problem_fullpath, verbose=False, animate=False, question_partial_plan=False):

    strictly_lower_cost = False

    time_start = time.time()

    # get MAPF solution to original problem (with original obstacles)
    raw_problem = explanations_multi.parse_yaml(problem_fullpath)
    solved = explanations_multi.generate_cbs_solution(problem_fullpath)
    if not solved:
        print('Cannot solve original problem')
        return False, []
    raw_solution = explanations_multi.parse_yaml(SOLUTION_YAML)
    raw_obstacles = raw_problem['map']['obstacles']
    raw_map = set([(obs[0], obs[1]) for obs in raw_obstacles])

    # Handling desired path of the agent and get agent name
    desired_paths = []
    agent_names = []
    for agent in raw_problem['agents']:
        if agent.get('waypoints') is not None:
            desired_paths.append(agent['waypoints'])
            agent_names.append(agent['name'])

    # no-obstacle zones
    forbidden_cells = []
    if question_partial_plan:
        # our agents' desired paths and all agents' start/goal
        for agent in raw_problem['agents']:
            forbidden_cells.append( agent['start'] )
            forbidden_cells.append( agent['goal'] )
        for desired_path in desired_paths:
            forbidden_cells += desired_path
    else:
        # our agents' desired paths and other agents' full paths
        for a in raw_solution['schedule']:
            if a not in agent_names:
                for pos in raw_solution['schedule'][a]:
                    forbidden_cells.append([pos['x'], pos['y']])
        for desired_path in desired_paths:
            forbidden_cells += desired_path

    # maps seen so far
    maps_seen_so_far = []
    maps_seen_so_far.append(raw_map)

    # queue of map changes
    Q = []
    Q.append((0, raw_problem['map']['obstacles']))

    # loop
    while len(Q) > 0:

        if time.time() - time_start > MAX_TIME:
            print('Time budget is over!!')
            break

        next = Q.pop(0)
        changes = next[0]
        new_obstacles = next[1]
        new_map = set([(obs[0], obs[1]) for obs in new_obstacles])
        print('--- next iteration: %d changes' % changes)

        # solve problem with this map
        new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, new_obstacles)
        new_filename = "additional/build/new_problem.yaml"
        explanations_multi.create_problem_yaml(new_problem, new_filename)
        solved = explanations_multi.generate_cbs_solution(ROOT_PATH + "/" + new_filename)
        if not solved:
            print('Problem not solved...')
            continue
        new_solution = explanations_multi.parse_yaml(SOLUTION_YAML)

        # check if solution satisfies desired paths (or has the same cost as if you forced agents to take desired paths)
        if question_partial_plan:
            success = explanations_multi.does_mapf_solution_satisfy_desired_paths(new_problem, new_solution, agent_names, desired_paths)
        else:
            success = explanations_multi.does_mapf_solution_satisfy_desired_paths_with_other_agents_fixed(new_problem, raw_solution, new_solution, agent_names, desired_paths)
        if success:
            print('Found solution to invmap!!')
            if animate:
                explanations_multi.generate_animation(raw_problem, new_problem, new_solution)

                solution_constrained = explanations_multi.parse_yaml(SOLUTION_YAML)
                explanations_multi.generate_animation(raw_problem, new_problem, solution_constrained)
            return True, new_obstacles

        # agents' cells (we can add obstacles to locations where any agent is passing on currently optimal solution)
        agent_cells = []
        for agent in new_solution['schedule']:
            for pos in new_solution['schedule'][agent]:
                agent_cells.append([pos['x'], pos['y']])

        # pruning
        #if not question_partial_plan:
        #    # if we need the full desired plan to be optimal, and our cost is already larger, we can never make it lower by adding more obstacles, therefore stop expanding this node
        #    if new_solution['statistics']['cost'] > raw_solution['statistics']['cost']:
        #        print('Cost is already larger...')
        #        continue

        # otherwise try out all combinations of obstacles to add
        for i in range(new_problem['map']['dimensions'][0]):
            for j in range(new_problem['map']['dimensions'][1]):
                if [i,j] not in new_obstacles and [i,j] not in forbidden_cells and [i,j] in agent_cells:
                    next_obstacles = copy.deepcopy(new_obstacles)
                    next_obstacles.append([i,j])
                    next_map = set([(obs[0], obs[1]) for obs in next_obstacles])
                    if next_map not in maps_seen_so_far:
                        Q.append( (changes+1, next_obstacles) )
                        maps_seen_so_far.append(next_map)
        Q.sort()

    # did not find a valid solution
    print('Could not find a valid solution to invmapf (queue exhausted)')
    return False, []


if __name__ == '__main__':
    # Terminal Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_file", help="input problem filepath")
    parser.add_argument("-v", "--verbose", action="store_true", help="outputs debug information")
    parser.add_argument("-a", "--animate", action="store_true", help="shows animation")
    parser.add_argument("-q", "--question-partial-plan", action="store_true", help="solves question for partial (instead of full) plan, i.e. 'why do agents A not take paths D?' instead of 'why not full plan X?'")
    args = parser.parse_args()
    # Main SP Function
    main_inv_mapf(args.problem_file, args.verbose, args.animate, args.question_partial_plan)
