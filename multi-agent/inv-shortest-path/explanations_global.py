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
import explanations_multi_incremental
import baseline_joint_single_agent
import copy
import time
import pdb

# NOTE: it seems that the most important thing is to have many possible "desired solutions" from the start (those gathered throughout the process do not seem to allow to solve more problems)
#       so how about obtaining all optimal ISP solutions for each constrained-agent separately, and perhaps all optimal baseline invmapf solutions (from joint single invmapf),
#       and then on those maps obtain the full constrained MAPF plan... then use these as possible desired plans....... could possibly improve #obst over joint-single-invmapf or incr solutions

MAX_TIME = 60

def main_inv_mapf(problem_file, verbose=False, animate=False, question_partial_plan=False):
    problem_fullpath = EXAMPLES_PATH + "/" + problem_file
    return main_inv_mapf_fullpath(problem_fullpath, verbose, animate, question_partial_plan)


def main_inv_mapf_fullpath(problem_fullpath, verbose=False, animate=False, question_partial_plan=False):

    # NOTE: what i want is probably to collect potential desired_solutions and bad_solutions; in each cycle i get both new desired_solutions and new bad_solutions
    # so i can have like two buckets, one with potential desired_solutions (where desired_paths satisfied) and one with bad solutions (where not satisfied)
    # at each iteration i obtain all ("N") obstacle combs that lead each of the desired_solutions to become lower-cost than all bad_solutions
    # iteration 1: 1 desired_solution , 1 bad_solution ---> N1 desired_solution , N1 bad_solution
    # iteration 2: 1+N1 desired_solution , 1+N1 bad_solution ---> (1+N1)*(N2) desired_solution , (1+N1)*(N2) bad_solution
    # iteration 3: ...
    # maybe desired_solutions could be put into a queue instead of bucket, once we understand we cannot solve with first desired_solutions we try with second on list

    strictly_lower_cost = False
    use_all_solutions = False    # does not seem have any impact on #problems solved or #changes (thought it increases/decreases computation time depending on problem)
    find_global_optimum = True   # necessary to find fewer-change maps than 'incr'

    time_start = time.time()

    # get MAPF solution to original problem (with original obstacles)
    raw_problem = explanations_multi.parse_yaml(problem_fullpath)
    solved = explanations_multi.generate_cbs_solution(problem_fullpath)
    if not solved:
        print('Cannot solve original problem')
        return False, []
    raw_solution = explanations_multi.parse_yaml(SOLUTION_YAML)

    # Handling desired path of the agent and get agent name
    desired_paths = []
    agent_names = []
    for agent in raw_problem['agents']:
        if agent.get('waypoints') is not None:
            desired_paths.append(agent['waypoints'])
            agent_names.append(agent['name'])

    # bucket of bad mapf solutions (where desired paths not met)
    bad_mapf_solutions = [ raw_solution ]

    # queue of desired_mapf_solutions (where desired paths met)
    Q = []

    # first desired mapf solution is mapf(obst=obstacles, constr=desired_paths)
    if question_partial_plan:
        solved = explanations_multi.generate_cbs_solution_constrained(problem_fullpath)
        if not solved:
            print('Cannot solve original problem s.t. desired paths')
            return False, []
        Q.append(explanations_multi.parse_yaml(SOLUTION_YAML))

    # second desired mapf solution is equal to original solution and we just replace our agents' paths (same as explanations_multi and explanations_multi_incremental)
    simple_desired_mapf_solution = copy.deepcopy(raw_solution)
    for i in range(len(desired_paths)):
        agent = agent_names[i]
        simple_desired_mapf_solution['schedule'][agent] = []
        for t, pos in enumerate(desired_paths[i]):
            simple_desired_mapf_solution['schedule'][agent].append( {'x':pos[0], 'y':pos[1], 't':t} )
    Q.append(simple_desired_mapf_solution)

    # more desired mapf solutions from joint-single baseline maps
    bsj_all_obstacle_solutions = None

    # loop to find inv-mapf solutions for 
    inv_mapf_solutions = []
    best_inv_mapf_solution_cost = 1e10
    while True:

        if (not find_global_optimum) and len(inv_mapf_solutions) > 0:
            break

        if time.time() - time_start > MAX_TIME:
            break

        # if we have ran out of desired mapf solutions, try using those from joint-single-baseline maps [here we compute the maps]
        if question_partial_plan and len(Q) == 0 and bsj_all_obstacle_solutions == None:
            bsj_success, bsj_all_obstacle_solutions = baseline_joint_single_agent.main_inv_mapf_fullpath(problem_fullpath, question_partial_plan=True, find_all_solutions=True)

        if time.time() - time_start > MAX_TIME:
            break

        # if we have ran out of desired mapf solutions, try using those from joint-single-baseline maps [here we compute the respective constrained mapf solutions]
        if question_partial_plan and len(Q) == 0 and bsj_all_obstacle_solutions != None and len(bsj_all_obstacle_solutions) > 0:
            while len(bsj_all_obstacle_solutions) > 0:
                sol_obstacles = bsj_all_obstacle_solutions.pop(0)
                # solve constrained MAPF problem on this map
                new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, sol_obstacles)
                new_filename = "additional/build/new_problem.yaml"
                explanations_multi.create_problem_yaml(new_problem, new_filename)
                solved = explanations_multi.generate_cbs_solution_constrained(ROOT_PATH + "/" + new_filename)
                if not solved:
                    continue
                new_desired_mapf_solution = explanations_multi.parse_yaml(SOLUTION_YAML)
                Q.append(new_desired_mapf_solution)
                break

        if len(Q) == 0:
            break

        desired_mapf_solution = Q.pop(0)

        new_obstacles = raw_problem['map']['obstacles']
        prev_obstacle_solutions = []

        # loop until desired solution is satisfied (or we understand there is no solution)
        found_map_for_this_desired_solution = False
        while not found_map_for_this_desired_solution:

            if use_all_solutions:
                # get all solutions to inv-mapf, i.e. all combinations of obstacles that lead cost(desired_mapf_solution) <= cost(bad_mapf_solutions)
                success, all_obstacle_solutions = explanations_multi_incremental.inv_mapf_incremental(raw_problem, raw_solution, new_obstacles, bad_mapf_solutions, desired_paths, agent_names, strictly_lower_cost, input_desired_mapf_solution=desired_mapf_solution, find_all_solutions=True)
            else:
                # get solution to inv-mapf, i.e. obstacles that lead cost(desired_mapf_solution) <= cost(bad_mapf_solutions)
                success, new_obstacles = explanations_multi_incremental.inv_mapf_incremental(raw_problem, raw_solution, new_obstacles, bad_mapf_solutions, desired_paths, agent_names, strictly_lower_cost, input_desired_mapf_solution=desired_mapf_solution, find_all_solutions=False)
                all_obstacle_solutions = [new_obstacles]

            if success and len(all_obstacle_solutions[0]) < best_inv_mapf_solution_cost:
                if all_obstacle_solutions == prev_obstacle_solutions:
                    break
                # for each map
                for sol_obstacles in all_obstacle_solutions:
                    # solve problem with this map
                    new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, sol_obstacles)
                    new_filename = "additional/build/new_problem.yaml"
                    explanations_multi.create_problem_yaml(new_problem, new_filename)
                    solved = explanations_multi.generate_cbs_solution(ROOT_PATH + "/" + new_filename)
                    if not solved:
                        continue
                    new_solution = explanations_multi.parse_yaml(SOLUTION_YAML)
                    # check if our desired_paths are possible in this map
                    solved_constrained = explanations_multi.generate_cbs_solution_constrained(ROOT_PATH + "/" + new_filename)
                    if not solved_constrained:
                        continue
                    # check if our desired_paths are optimal in this map
                    new_solution_constrained = explanations_multi.parse_yaml(SOLUTION_YAML)
                    if question_partial_plan:
                        is_solution = explanations_multi.does_mapf_solution_satisfy_desired_paths(new_problem, new_solution, agent_names, desired_paths)
                    else:
                        is_solution = explanations_multi.does_mapf_solution_satisfy_desired_paths_with_other_agents_fixed(new_problem, raw_solution, new_solution, agent_names, desired_paths)
                    #if new_solution['statistics']['cost'] == new_solution_constrained['statistics']['cost']:
                    if is_solution:
                        # found invmapf solution! (not sure it is the lowest cost, though)
                        inv_mapf_solutions.append( [new_problem, new_solution_constrained] )
                        found_map_for_this_desired_solution = True
                        if len(sol_obstacles) < best_inv_mapf_solution_cost:
                            best_inv_mapf_solution_cost = len(sol_obstacles)
                        if not find_global_optimum:
                            break
                    else:
                        # found new desired_solution (a mapf solution where desired_paths hold)
                        if question_partial_plan and find_global_optimum:
                            Q.append(new_solution_constrained) # ----> adding these extra solutions is necessary to obtain fewer-change maps
                        # found a new bad_solution (a mapf solution where desired_paths do not hold)
                        bad_mapf_solutions.append(new_solution)
                # next obstacles
                new_obstacles = all_obstacle_solutions[0] # TODO: could consider trying to solve with each of the obstacle combinations
            else:
                break

            prev_obstacle_solutions = all_obstacle_solutions

    # exhausted queue so find the best inv-mapf solution
    if len(inv_mapf_solutions) > 0:
        costs = []
        for sol in inv_mapf_solutions:
            costs.append( len(sol[0]['map']['obstacles']) )
        best_sol = inv_mapf_solutions[costs.index(min(costs))]
        # animate and return
        if animate:
            explanations_multi.generate_animation(raw_problem, best_sol[0], best_sol[1])
        return True, best_sol[0]['map']['obstacles']
    else:
        return False, []


def main_inv_mapf_fullpath_old(problem_fullpath, verbose=False, animate=False):

    # old version
    return False, []

    strictly_lower_cost = False

    time_start = time.time()

    # get MAPF solution to original problem (with original obstacles)
    raw_problem = explanations_multi.parse_yaml(problem_fullpath)
    solved = explanations_multi.generate_cbs_solution(problem_fullpath)
    if not solved:
        print('Cannot solve original problem')
        return False, []
    raw_solution = explanations_multi.parse_yaml(SOLUTION_YAML)

    # Handling desired path of the agent and get agent name
    desired_paths = []
    agent_names = []
    for agent in raw_problem['agents']:
        if agent.get('waypoints') is not None:
            desired_paths.append(agent['waypoints'])
            agent_names.append(agent['name'])

    # first sequence
    first_seq = [ [raw_problem, raw_solution] ]

    # inv-mapf solutions
    inv_mapf_solutions = []

    # queue
    Q = []
    Q.append(first_seq)
    while len(Q) > 0:

        if time.time() - time_start > MAX_TIME:
            break

        seq = Q.pop(0)

        # get obstacles
        new_obstacles = seq[-1][0]['map']['obstacles']

        # desired mapf solution is mapf(obst=seq[-1].obstacles, constr=desired_paths)
        new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, new_obstacles)
        new_filename = "additional/build/new_problem.yaml"
        explanations_multi.create_problem_yaml(new_problem, new_filename)
        solved = explanations_multi.generate_cbs_solution_constrained(ROOT_PATH + "/" + new_filename)
        if not solved:
            continue
        desired_mapf_solution = explanations_multi.parse_yaml(SOLUTION_YAML)

        # sanity check constrained solution is working
        for i in range(len(desired_paths)):
            agent = agent_names[i]
            desired_path = desired_paths[i]
            path = desired_mapf_solution['schedule'][agent]
            if len(desired_path) != len(path):
                print('ERROR: desired path not met')
                #pdb.set_trace()
                break
            for t in range(len(desired_path)):
                if tuple(desired_path[t]) != (path[t]['x'], path[t]['y']):
                    print('ERROR: desired path not met')
                    #pdb.set_trace()
                    break

        # bad mapf solutions
        bad_mapf_solutions = [ pair[1] for pair in seq  ]

        print('-------------------------- next in the queue (%d obstacles vs %s originally)' % (len(new_obstacles), len(raw_problem['map']['obstacles'])))
        print('----- size of sequence: %d' % len(seq))
        print('----- number of bad mapf solutions: %d' % len(bad_mapf_solutions))
        print('----- size of queue: %d' % len(Q))

        # get all solutions to inv-mapf, i.e. all combinations of obstacles that lead cost(desired_mapf_solution) <= cost(bad_mapf_solutions)
        success, all_obstacle_solutions = explanations_multi_incremental.inv_mapf_incremental(raw_problem, raw_solution, new_obstacles, bad_mapf_solutions, desired_paths, agent_names, strictly_lower_cost, input_desired_mapf_solution=desired_mapf_solution, find_all_solutions=True)
        if success:
            # for each map
            for sol_obstacles in all_obstacle_solutions:
                # solve problem with this map
                new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, sol_obstacles)
                new_filename = "additional/build/new_problem.yaml"
                explanations_multi.create_problem_yaml(new_problem, new_filename)
                solved = explanations_multi.generate_cbs_solution(ROOT_PATH + "/" + new_filename)
                if not solved:
                    continue
                new_solution = explanations_multi.parse_yaml(SOLUTION_YAML)
                # check if our desired_paths are possible in this map
                solved_constrained = explanations_multi.generate_cbs_solution_constrained(ROOT_PATH + "/" + new_filename)
                if not solved_constrained:
                    continue
                # check if our desired_paths are optimal in this map
                new_solution_constrained = explanations_multi.parse_yaml(SOLUTION_YAML)
                cost_new_sol = 0
                for agent in new_solution['schedule']:
                    cost_new_sol += len(new_solution['schedule'][agent]) - 1
                cost_new_sol_constrained = 0
                for agent in new_solution['schedule']:
                    cost_new_sol_constrained += len(new_solution_constrained['schedule'][agent]) - 1
                sol_optimal = (cost_new_sol == cost_new_sol_constrained)
                if sol_optimal:
                    inv_mapf_solutions.append( [new_problem, new_solution_constrained] )
                    continue
                # add obstacles/badpaths to queue
                new_seq = seq + [ [new_problem, new_solution] ]
                Q.append(new_seq)

    #pdb.set_trace()

    # exhausted queue so find the best inv-mapf solution
    if len(inv_mapf_solutions) > 0:
        costs = []
        for sol in inv_mapf_solutions:
            costs.append( len(sol[0]['map']['obstacles']) )
        best_sol = inv_mapf_solutions[costs.index(min(costs))]
        # animate and return
        if animate:
            explanations_multi.generate_animation(raw_problem, best_sol[0], best_sol[1])
        return True, best_sol[0]['map']['obstacles']
    else:
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
