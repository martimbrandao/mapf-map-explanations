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

SAVE_FIGURES = False

global_problem_fullpath = ''


def inv_mapf_incremental(raw_problem, raw_solution, new_obstacles, bad_mapf_solutions, input_desired_paths, agent_names, strictly_lower_cost=False, obst_over_desired_sol=False, input_desired_mapf_solution=None, find_all_solutions=False, verbose=False):

    # desired solution
    if input_desired_mapf_solution == None:
        # if no desired solution was given, assume desired_paths with other agents fixed
        desired_paths = copy.deepcopy(input_desired_paths)
        desired_mapf_solution = copy.deepcopy(raw_solution)
        for i in range(len(desired_paths)):
            agent = agent_names[i]
            desired_mapf_solution['schedule'][agent] = []
            for t, pos in enumerate(desired_paths[i]):
                desired_mapf_solution['schedule'][agent].append( {'x':pos[0], 'y':pos[1], 't':t} )
    else:
        # desired solution given as input
        desired_mapf_solution = copy.deepcopy(input_desired_mapf_solution)
        desired_paths = []
        for agent in agent_names:
            desired_path = [ [pos['x'], pos['y']] for pos in desired_mapf_solution['schedule'][agent] ]
            desired_paths.append(desired_path)

    # TODO v1: first find our ALL possible combinations of obstacles that can be put in order to make cost[desired_paths] < cost[bad_solutions_for_desired_agents_only]
    # then, use those combinations to generate potential desired_mapf_solutions[i] = mapf(obst=comb_i, constr=desired_paths), now considering all agents
    # for each of those potential solutions, try to obtain solution (new obstacles)
    # take the solution with minimum obstacles
    #
    # TODO v2: first find our ALL possible combinations of obstacles that can be put in order to make cost[desired_paths] < cost[bad_solutions_for_desired_agents_only]
    # find mapf solutions where other agents go through our new obstacles (therefore serving as obstacles)
    #
    #new_filename = "additional/build/new_problem.yaml"
    #
    # get optimal solution for mapf(obst=raw_obstacles, constraint=desired_paths)
    #explanations_multi.create_problem_yaml(raw_problem, new_filename)
    #solved = generate_cbs_solution_constrained(ROOT_PATH + "/" + new_filename)
    #if not solved:
    #    print('Could not solve mapf(obst=raw_obstacles, constraint=desired_paths)')
    #    # did not work so we try to get optimal solution for mapf(obst=new_obstacles, constraint=desired_paths) 
    #    new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, new_obstacles)
    #    explanations_multi.create_problem_yaml(new_problem, new_filename)
    #    solved = generate_cbs_solution_constrained(ROOT_PATH + "/" + new_filename)
    #    if not solved:
    #        print('Could not solve mapf(obst=new_obstacles, constraint=desired_paths)')
    #        return False, []
    #
    # TODO: find all/diverse optimal solutions for mapf(obst=raw_obstacles, constraint=desired_paths)
    #       see if we can make any of those lower cost than current

    # we have the full MAPF solution we want to make optimal now
    #desired_mapf_solution = explanations_multi.parse_yaml(SOLUTION_YAML)

    # setup
    graph = explanations_multi.create_graph(raw_problem)
    ori_makespan = copy.deepcopy(desired_mapf_solution['statistics']['makespan'])
    schedule = copy.deepcopy(desired_mapf_solution['schedule'])
    for agent_name in agent_names:
        schedule.pop(agent_name)

    # check validity of desired path
    if True:
        for path in schedule.values():
            for t, pos in enumerate(path):
                n = (pos['x'], pos['y'])
                for desired_path in desired_paths:
                    if n == tuple(desired_path[min(t, len(desired_path) - 1)]):
                        print("INVALID DESIRED PATH - Desired path of agent collides with other agents (same cell)")
                        pdb.set_trace()
                        return False, []
            for desired_path in desired_paths:
                for t in range(min(len(path)-1, len(desired_path)-1)):
                    p0 = (path[t]['x'], path[t]['y'])
                    p1 = (path[t+1]['x'], path[t+1]['y'])
                    d0 = tuple(desired_path[t])
                    d1 = tuple(desired_path[t+1])
                    if p0 == d1 and d0 == p1:
                        print("INVALID DESIRED PATH - Desired path of agent collides with other agents (agent swap)")
                        pdb.set_trace()
                        return False, []
        for desired_path in desired_paths:
            for t in range(ori_makespan, len(desired_path)):
                for path in schedule.values():
                    last_node = (path[-1]['x'], path[-1]['y'])
                    if tuple(desired_path[t]) == last_node:
                        print("INVALID DESIRED PATH - Desired path of agent collides with other agents (goal cell)")
                        return False, []

    # Determine max t
    max_t = 0
    for desired_path in desired_paths:
        max_t = max(len(desired_path) - 1, ori_makespan, max_t)

    ##################################
    ###
    ### optimization
    ###
    ### goal: find new obstacles such that, for all i, cost[bad_mapf_solution_i] > cost[desired_mapf_solution]

    # l_original
    l_original = []
    for i, n in enumerate(graph.nodes):
        if graph.nodes[n].get('obstacle'):
            l_original.append(1)
        else:
            l_original.append(0)

    # Auxiliary variables
    edges = []
    edge_t_2idx = {}  # key: tuple[edge, t] where t=0 means edge going from time=0 to time=1
    for t in range(max_t):
        for i, n in enumerate(graph.nodes):
            next_nodes = [nei for nei in graph[n]]
            next_nodes.append(n)
            for next_node in next_nodes:
                edge = (n, next_node)
                idx = len(edges)
                edge_t_2idx[(edge, t)] = idx
                edges.append(edge)
    cells = []
    edge2lidx = {}
    node2lidx = {}
    for i, n in enumerate(graph.nodes):
        cells.append(n)
        node2lidx[n] = i
        edge2lidx[(n, n)] = i
        for nei in graph[n]:
            edge = (nei, n)
            edge2lidx[edge] = i

    # Variables
    l_ = cp.Variable(len(l_original), boolean=True)

    # Cost
    cost = cp.norm1(l_ - l_original)

    # Constraints
    constraints = []

    # Constraints [G @ l_ <= h]  (v1.0)
    if False:
      #     sum_(j in good_path) 1000 * l_j + 1 <= sum_(j in bad_path) 1000 * l_j + 1
      # so: sum_(j in good_path) 1000 * l_j - sum_(j in bad_path) 1000 * l_j <= sum_(j in bad_path) 1 - sum_(j in good_path) 1
      G = []
      h = []
      for bad_sol in bad_mapf_solutions:
          Gline = [0.0] * len(l_original)
          hline = 0
          for agent in desired_mapf_solution['schedule']:
              for pos in desired_mapf_solution['schedule'][agent]:
                  j = node2lidx[ (pos['x'], pos['y']) ]
                  Gline[j] += 1000
                  hline -= 1
          for agent in bad_sol['schedule']:
              for pos in bad_sol['schedule'][agent]:
                  j = node2lidx[ (pos['x'], pos['y']) ]
                  Gline[j] -= 1000
                  hline += 1
          # NOTE: this is to force strict inequality "<" instead of "<=" (i.e. to force solution returned by CBS to be equal to ours, not just same cost)
          if strictly_lower_cost:
              hline -= 0.1
          G.append(Gline)
          h.append(hline)
      G = np.array(G)
      h = np.array(h)
      constraints.append(G @ l_ <= h)

    # Constraints [sum_(j in bad_path) l_j >= 1]  (v2.0)
    if True:
      G = []
      h = []
      for bad_sol in bad_mapf_solutions:
          Gline = [0.0] * len(l_original)
          for agent in bad_sol['schedule']:
              for pos in bad_sol['schedule'][agent]:
                  j = node2lidx[ (pos['x'], pos['y']) ]
                  Gline[j] += 1
          G.append(Gline)
          h.append(1)
      G = np.array(G)
      h = np.array(h)
      constraints.append(G @ l_ >= h)

    # Constraints l(desired_mapf_solution) = 0, i.e. don't add obstacles to cells used by our desired solution
    if obst_over_desired_sol:
        for agent in desired_mapf_solution['schedule']:
            pos = desired_mapf_solution['schedule'][agent][0]
            j = node2lidx[ (pos['x'], pos['y']) ]
            constraints.append(l_[j] == 0)
            pos = desired_mapf_solution['schedule'][agent][-1]
            j = node2lidx[ (pos['x'], pos['y']) ]
            constraints.append(l_[j] == 0)
        for agent in agent_names:
            for pos in desired_mapf_solution['schedule'][agent]:
                j = node2lidx[ (pos['x'], pos['y']) ]
                constraints.append(l_[j] == 0)
    else:
        for agent in desired_mapf_solution['schedule']:
            for pos in desired_mapf_solution['schedule'][agent]:
                j = node2lidx[ (pos['x'], pos['y']) ]
                constraints.append(l_[j] == 0)

    # Solve with cvxpy
    prob = cp.Problem(cp.Minimize(cost), constraints)
    value = prob.solve(solver=cp.GUROBI)
    if value == float('inf'):
        print("inverse shortest path FAILED")
        return False, []

    # New obstacles set
    new_obstacles = []
    for i, v in enumerate(l_.value):
        if round(v) == 1:
            new_obstacles.append(list(cells[i]))
    print('Found solution to optimization problem')

    # sanity check
    if sorted(new_obstacles) == sorted(raw_problem['map']['obstacles']):
        print('WARNING: inv-mapf-incremental did not give us new obstacles...')
        #pdb.set_trace()

    # if converged then enumerate all optimal solutions
    if find_all_solutions:
        # https://www.ibm.com/support/pages/using-cplex-examine-alternate-optimal-solutions
        # Let x{S} be the binary variables. Suppose you have a binary solution x* in available from the most recent optimization. Let N be the subset of S such that x*[n] = 1 for all n in N
        # Then, add the following constraint:
        # sum{n in N} x[n] - sum{s in S\N} x[s] <= |N|-1
        all_solutions = [l_.value]
        constraint_prev_sols = []
        constraint_same_l1 = (cp.norm1(l_ - l_original) <= prob.value)
        while True:
            print('Obtained %d optimal solutions...' % len(all_solutions))
            diff = (l_.value==1) * 1 - (l_.value==0) * 1
            constraint_prev_sols.append( diff @ l_ <= np.sum(l_.value==1) - 1 )
            prob = cp.Problem(cp.Minimize(cost), constraints + [constraint_same_l1] + constraint_prev_sols)
            res = prob.solve(solver=cp.GUROBI)
            if res == float('inf'):
                print('Number of optimal solutions: %d' % len(all_solutions))
                all_solution_obstacles = []
                for sol in all_solutions:
                    sol_obstacles = []
                    for i, v in enumerate(sol):
                        if round(v) == 1:
                            sol_obstacles.append(list(cells[i]))
                    all_solution_obstacles.append(sol_obstacles)
                return True, all_solution_obstacles
            all_solutions.append(l_.value)

    return True, new_obstacles


def main_inv_mapf(problem_file, verbose=False, animate=False, question_partial_plan=False, find_all_solutions=False):
    problem_fullpath = EXAMPLES_PATH + "/" + problem_file
    return main_inv_mapf_fullpath(problem_fullpath, verbose, animate, question_partial_plan, find_all_solutions)


def main_inv_mapf_fullpath(problem_fullpath, verbose=False, animate=False, question_partial_plan=False, find_all_solutions=False):

    global global_problem_fullpath
    global_problem_fullpath = problem_fullpath

    # Parsing and generating CBS solution of original problem file
    raw_problem = explanations_multi.parse_yaml(problem_fullpath)
    solved = explanations_multi.generate_cbs_solution(problem_fullpath)
    if not solved:
        print('Cannot solve original problem')
        return False, []
    raw_solution = explanations_multi.parse_yaml(SOLUTION_YAML)

    # Find MAPF plan := mapf(orig_map) s.t. desired_paths
    solution_constrained = None
    if question_partial_plan:
        solved = explanations_multi.generate_cbs_solution_constrained(problem_fullpath)
        if not solved:
            print('Cannot solve original problem s.t. desired paths')
            return False, []
        solution_constrained = explanations_multi.parse_yaml(SOLUTION_YAML)

    # straightforward InvMAPF with fixed paths:
    # find a map that leads to the MAPF plan := all agents as before but our agents with desired paths
    return main_inv_mapf_prob(raw_problem, raw_solution, obst_over_desired_sol=False, verbose=verbose, animate=animate, question_partial_plan=question_partial_plan, desired_mapf_solution=solution_constrained, find_all_solutions=find_all_solutions)

    #if question_partial_plan:
    #    # try without letting obstacles over desired mapf sol
    #    success, new_obstacles = main_inv_mapf_prob(raw_problem, raw_solution, obst_over_desired_sol=False, verbose=verbose, animate=animate, question_partial_plan=question_partial_plan, desired_mapf_solution=solution_constrained, find_all_solutions=find_all_solutions)
    #    if success:
    #        return success, new_obstacles
    #    # now try while letting obstacles over desired mapf sol
    #    return main_inv_mapf_prob(raw_problem, raw_solution, obst_over_desired_sol=True, verbose=verbose, animate=animate, question_partial_plan=question_partial_plan, desired_mapf_solution=solution_constrained, find_all_solutions=find_all_solutions)        
    #else:
    #    return main_inv_mapf_prob(raw_problem, raw_solution, obst_over_desired_sol=False, verbose=verbose, animate=animate, question_partial_plan=question_partial_plan, desired_mapf_solution=solution_constrained, find_all_solutions=find_all_solutions)


def main_inv_mapf_prob(raw_problem, raw_solution, obst_over_desired_sol=False, verbose=False, animate=False,  question_partial_plan=False, desired_mapf_solution=None, find_all_solutions=False):

    strictly_equal_solution = False

    # Handling desired path of the agent and get agent name
    desired_paths = []
    agent_names = []
    for agent in raw_problem['agents']:
        if agent.get('waypoints') is not None:
            desired_paths.append(agent['waypoints'])
            agent_names.append(agent['name'])

    # desired paths already optimal
    desired_paths_already_optimal = True
    for i in range(len(agent_names)):
        for desired_path in desired_paths:
            if desired_path[i] != raw_solution['schedule'][agent_names[i]]:
                desired_paths_already_optimal = False
    if desired_paths_already_optimal:
        print('Desired paths already optimal')
        return True, []
    if desired_mapf_solution != None and raw_solution['statistics']['cost'] == desired_mapf_solution['statistics']['cost']:
        print('Desired paths already optimal')
        return True, []

    # initial obstacles
    new_obstacles = copy.deepcopy(raw_problem['map']['obstacles'])

    # mapf solutions that do not meet our desired paths (hence should be made higher cost)
    bad_mapf_solutions = []
    bad_mapf_solutions.append(copy.deepcopy(raw_solution))

    # do we enforce strictly lower costlower cost of desired solutions?
    strictly_lower_cost = strictly_equal_solution

    # loop until desired path is in optimal solution
    n_iterations = 0
    while True:

        # limit iterations
        if n_iterations > 50:
            print('Could not solve problem within max-iterations')
            return False, []
        n_iterations += 1

        # obtain obstacles that make, for all i, cost[bad_mapf_solution_i] >= cost[mapf(obst=new_obstacles, constraint=desired_paths)]
        success, new_obstacles2 = inv_mapf_incremental(raw_problem, raw_solution, new_obstacles, bad_mapf_solutions, desired_paths, agent_names, strictly_lower_cost, obst_over_desired_sol, desired_mapf_solution, verbose=verbose)
        if not success:
            print('Could not solve inv-mapf-incremental, i.e. no way to place obstacles such that cost[bad_sol] > cost[desired_sol]')
            return False, []

        # solve this new problem (with new obstacles)
        new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, new_obstacles2)
        new_filename = "additional/build/new_problem.yaml"
        explanations_multi.create_problem_yaml(new_problem, new_filename)
        solved = explanations_multi.generate_cbs_solution(ROOT_PATH + "/" + new_filename)
        if not solved:
            print('Could not solve new problem (with new obstacles)')
            return False, []
        new_solution = explanations_multi.parse_yaml(SOLUTION_YAML)

        # check that this solution has the same cost as the desired solution
        if not strictly_equal_solution:
            if question_partial_plan:
                success = explanations_multi.does_mapf_solution_satisfy_desired_paths(new_problem, new_solution, agent_names, desired_paths)
            else:
                success = explanations_multi.does_mapf_solution_satisfy_desired_paths_with_other_agents_fixed(new_problem, raw_solution, new_solution, agent_names, desired_paths)
            if success:
                break

        # check desired paths already optimal (strictly equal case)
        desired_paths_already_optimal = True
        for i in range(len(desired_paths)):
            agent = agent_names[i]
            path = new_solution['schedule'][agent]
            if len(path) != len(desired_paths[i]):
                desired_paths_already_optimal = False
                break
            for t, pos in enumerate(path):
                if tuple(desired_paths[i][t]) != (pos['x'], pos['y']):
                    desired_paths_already_optimal = False
                    break
        if desired_paths_already_optimal:
            break


        # animation for debug
        #print('Showing desired plan on new map')
        #desired_mapf_solution2 = copy.deepcopy(raw_solution)
        #for i in range(len(desired_paths)):
        #    agent = agent_names[i]
        #    desired_mapf_solution2['schedule'][agent] = []
        #    for t, pos in enumerate(desired_paths[i]):
        #        desired_mapf_solution2['schedule'][agent].append( {'x':pos[0], 'y':pos[1], 't':t} )
        #desired_mapf_solution2['statistics']['cost'] = 0
        #for a in desired_mapf_solution2['schedule']:
        #    path = desired_mapf_solution2['schedule'][a]
        #    desired_mapf_solution2['statistics']['cost'] += len(path) - 1
        #explanations_multi.generate_animation(raw_problem, new_problem, desired_mapf_solution)
        #pdb.set_trace()
        #
        #print('Showing new solution for new map')
        #explanations_multi.generate_animation(raw_problem, new_problem, new_solution)


        # if we are not making any progress then try to enforce strictly lower cost of our solution
        if new_solution['schedule'] == bad_mapf_solutions[-1]['schedule'] and strictly_lower_cost == False:
            strictly_lower_cost = True

        # this solution is a bad solution, i.e. its cost must be made higher than the cost with desired_paths
        bad_mapf_solutions.append(copy.deepcopy(new_solution))
        new_obstacles = new_obstacles2

    # solution found!
    print('Success! Found inv-mapf solution')

    #print('Showing new solution for new map')
    #explanations_multi.generate_animation(raw_problem, new_problem, new_solution)

    # check all or single solution depending on what we want
    if find_all_solutions:

        success, all_solution_obstacles = inv_mapf_incremental(raw_problem, raw_solution, new_obstacles, bad_mapf_solutions, desired_paths, agent_names, strictly_lower_cost, obst_over_desired_sol, desired_mapf_solution, find_all_solutions=True)
        good_solution_obstacles = []
        good_solutions = []
        for sol_obstacles in all_solution_obstacles:
            # check if this solution passes sanity check
            new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, sol_obstacles)
            new_filename = "additional/build/new_problem.yaml"
            explanations_multi.create_problem_yaml(new_problem, new_filename)
            solved = explanations_multi.generate_cbs_solution(ROOT_PATH + "/" + new_filename)
            if not solved:
                continue
            new_solution = explanations_multi.parse_yaml(SOLUTION_YAML)
            if question_partial_plan:
                success = explanations_multi.does_mapf_solution_satisfy_desired_paths(new_problem, new_solution, agent_names, desired_paths)
            else:
                success = explanations_multi.does_mapf_solution_satisfy_desired_paths_with_other_agents_fixed(new_problem, raw_solution, new_solution, agent_names, desired_paths)
            if success:
                good_solution_obstacles.append(sol_obstacles)
                good_solutions.append([new_problem, new_solution])
        print('Found %d good solutions' % len(good_solution_obstacles))
        #if len(good_solution_obstacles) > 1:
        #    for sol in good_solutions:
        #        explanations_multi.generate_animation(raw_problem, sol[0], sol[1])
        #    pdb.set_trace()
        return True, good_solution_obstacles

    else:

        new_obstacles = new_obstacles2
        if question_partial_plan:
            success = explanations_multi.does_mapf_solution_satisfy_desired_paths(new_problem, new_solution, agent_names, desired_paths)
        else:
            success = explanations_multi.does_mapf_solution_satisfy_desired_paths_with_other_agents_fixed(new_problem, raw_solution, new_solution, agent_names, desired_paths)
        if not success:
            print('Oops, solution does not pass sanity check')
            return False, []
        if animate:
            print('Showing original plan on original map')
            explanations_multi.generate_animation(raw_problem, raw_problem, raw_solution)

            #print('Showing original plan on new map')
            #explanations_multi.generate_animation(raw_problem, new_problem, raw_solution)

            print('Showing desired plan on original map')
            desired_mapf_solution = copy.deepcopy(raw_solution)
            for i in range(len(desired_paths)):
                agent = agent_names[i]
                desired_mapf_solution['schedule'][agent] = []
                for t, pos in enumerate(desired_paths[i]):
                    desired_mapf_solution['schedule'][agent].append( {'x':pos[0], 'y':pos[1], 't':t} )
            explanations_multi.generate_animation(raw_problem, raw_problem, desired_mapf_solution)

            print('Showing desired plan on new map')
            explanations_multi.generate_animation(raw_problem, new_problem, desired_mapf_solution)

            print('Showing new solution on new map')
            explanations_multi.generate_animation(raw_problem, new_problem, new_solution)
            # debug animation
            #desired_solution = copy.deepcopy(raw_solution)
            #for i in range(len(desired_paths)):
            #    agent = agent_names[i]
            #    desired_solution['schedule'][agent] = []
            #    for t, pos in enumerate(desired_paths[i]):
            #        desired_solution['schedule'][agent].append( {'x':pos[0], 'y':pos[1], 't':t} )
            #explanations_multi.generate_animation(raw_problem, new_problem, desired_solution)

        if SAVE_FIGURES:
            print('Showing original plan on original map')
            explanations_multi.generate_animation(raw_problem, raw_problem, raw_solution, still=True, save=os.path.splitext(global_problem_fullpath)[0]+'oldmap-oldplan')

            print('Showing desired plan on original map')
            desired_mapf_solution = copy.deepcopy(raw_solution)
            for i in range(len(desired_paths)):
                agent = agent_names[i]
                desired_mapf_solution['schedule'][agent] = []
                for t, pos in enumerate(desired_paths[i]):
                    desired_mapf_solution['schedule'][agent].append( {'x':pos[0], 'y':pos[1], 't':t} )
            explanations_multi.generate_animation(raw_problem, raw_problem, desired_mapf_solution, still=True, save=os.path.splitext(global_problem_fullpath)[0]+'oldmap-desiredplan', show_waypoints=True)

            print('Showing desired plan on new map')
            explanations_multi.generate_animation(raw_problem, new_problem, desired_mapf_solution, still=True, save=os.path.splitext(global_problem_fullpath)[0]+'newmap-desiredplan', show_waypoints=True)

            print('Showing new solution on new map')
            explanations_multi.generate_animation(raw_problem, new_problem, new_solution, still=True, save=os.path.splitext(global_problem_fullpath)[0]+'newmap-newplan')

        return True, new_obstacles


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
