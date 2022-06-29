import yaml
import networkx as nx
import cvxpy as cp
import numpy as np
import argparse
import copy
import pdb
from path import *
import explanations_multi

# Parse YAML to create graph
def parse_yaml(filepath):
    # Parse YAML
    with open(filepath, 'r') as stream:
        try:
            raw_dict = yaml.safe_load(stream)
            print(filepath, "loaded successfully")
        except yaml.YAMLError as e:
            print(e)
    # Extract data
    dimensions = raw_dict['map']['dimensions']
    obstacles = raw_dict['map']['obstacles']
    # Create Graph
    graph = nx.Graph()
    for r in range(dimensions[0]):
        for c in range(dimensions[1]):
            # horizontal edges
            if c < dimensions[0] - 1:
                graph.add_edge((r, c), (r, c + 0.5))
                graph.add_edge((r, c + 0.5), (r, c + 1))
            # vertical edges
            if r < dimensions[1] - 1:
                graph.add_edge((r, c), (r + 0.5, c))
                graph.add_edge((r + 0.5, c), (r + 1, c))
    # Flag Portals and setup area types
    for n in graph.nodes:
        if isinstance(n[0], int) and isinstance(n[1], int):
            graph.nodes[n]['portal'] = False
            graph.nodes[n]['area_type'] = 0
        else:
            graph.nodes[n]['portal'] = True
    # Setup obstacles
    for obs in obstacles:
        graph.nodes[tuple(obs)]['area_type'] = 1
    # Return
    return graph, raw_dict


# Insert portal nodes to raw path, e.g. [(0,0), (0,1)] to [(0,0), (0,0.5), (0,1)]
def insert_portals(path):
    new_path = []
    for i in range(len(path) - 1):
        portal = ((path[i][0] + path[i + 1][0]) / 2, (path[i][1] + path[i + 1][1]) / 2)
        new_path.extend([path[i], portal])
    return new_path


# Delete portal nodes from output path, e.g. [(0,0), (0,0.5), (0,1)] to [(0,0), (0,1)]
def delete_portals(path):
    return [path[i] for i in range(len(path)) if i % 2 == 0]


# Calculate cost of given path in given graph
def get_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]["weight"]
    return cost


def create_aux_vars(graph, desired_paths):
    # Nodes, edges, non-portal nodes (varnodes)
    nodes = list(graph.nodes)
    edge2index = {}
    edges = []
    edge2varnodeindex = {}
    varnodes = []
    for (i, j) in graph.edges:
        edge2index[i, j] = len(edges)
        edges.append([i, j])
        edge2index[j, i] = len(edges)
        edges.append([j, i])
        if not graph.nodes[i]["portal"]:
            vn = i
        else:
            vn = j
        if vn in varnodes:
            idx = varnodes.index(vn)
        else:
            idx = len(varnodes)
            varnodes.append(vn)
        edge2varnodeindex[i, j] = idx
        edge2varnodeindex[j, i] = idx

    # A matrix
    A = np.zeros([len(nodes), len(edges)])
    for i in range(len(nodes)):
        for nei in graph.adj[nodes[i]]:
            j = edge2index[nodes[i], nei]
            A[i, j] = 1
            j = edge2index[nei, nodes[i]]
            A[i, j] = -1

    # insert portals to desired paths
    vec_dp = []
    for desired_path in desired_paths:
        dp = insert_portals(desired_path)
        vec_dp.append(dp)

    # Desired x
    vec_xzero = []
    for dp in vec_dp:
        xzero = np.zeros(len(edges))
        for p in range(len(dp) - 1):
            j = edge2index[dp[p], dp[p + 1]]
            xzero[j] = 1
        vec_xzero.append(xzero)

    # Return
    return vec_dp, nodes, edges, edge2index, varnodes, edge2varnodeindex, A, vec_xzero


# Sanity Check
def check(dp, new_graph):
    success = False
    new_path = nx.shortest_path(new_graph, source=dp[0], target=dp[-1], weight="weight")
    if get_cost(new_graph, new_path) == get_cost(new_graph, dp):
        success = True
        print("inverse shortest path: success")
    else:
        print("inverse shortest path: fail")
    return success


# ISP Discrete
def isp_discrete(graph, agent_names, desired_paths_input, area_costs=None, allowed_area_types=None, dist_per_edge=0.5, forced_free_locations=[], find_all_solutions=False):
    if allowed_area_types is None:
        allowed_area_types = [0, 1]
    if area_costs is None:
        area_costs = [1, 1000]

    desired_paths = []
    for dpi in desired_paths_input:
        desired_paths.append([(x[0], x[1]) for x in dpi])

    vec_dp, nodes, edges, edge2index, varnodes, edge2varnodeindex, A, vec_xzero = create_aux_vars(graph, desired_paths)

    # l_original
    l_original = np.zeros(len(varnodes) * len(allowed_area_types))
    for idx in range(len(varnodes)):
        node = varnodes[idx]
        for k in range(len(allowed_area_types)):
            if allowed_area_types[k] == graph.nodes[node]["area_type"]:
                l_original[len(allowed_area_types) * idx + k] = 1
            else:
                l_original[len(allowed_area_types) * idx + k] = 0

    # - inverse optimization problem -
    # Variables and constraints
    l_ = cp.Variable(len(l_original), boolean=True)
    vec_pi_ = []
    vec_lambda_ = []
    constraints = []
    for i_desired_path in range(len(desired_paths)):
        # add vars
        vec_pi_.append(cp.Variable(len(nodes)))
        vec_lambda_.append(cp.Variable(len(edges)))
        # current agent's vars
        pi_ = vec_pi_[-1]
        lambda_ = vec_lambda_[-1]
        desired_path = desired_paths[i_desired_path]
        dp = vec_dp[i_desired_path]
        xzero = vec_xzero[i_desired_path]
        # constraints
        for j in range(len(edges)):
            edge = edges[j]
            i = edge2varnodeindex[edge[0], edge[1]]
            # edge's new cost d_j = sum_(k in areas) dist_j * ac_k * l_ik
            d_j = 0
            for k in range(len(allowed_area_types)):
                ac_k = area_costs[allowed_area_types[k]]
                d_j += dist_per_edge * ac_k * l_[len(allowed_area_types) * i + k]
            if xzero[j] == 1:
                # sum_i a_ij * pi_i = d_j,              for all j in desired path
                constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) == d_j)
            else:
                # sum_i a_ij * pi_i + lambda_j = d_j,   for all j not in desired path
                constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) + lambda_[j] == d_j)
        # sum_k l_ik = 1, for all i
        for i in range(len(varnodes)):
            idx = len(allowed_area_types) * i
            constraints.append(cp.sum(l_[idx:idx + len(allowed_area_types)]) == 1)
        # lambda >= 0, for all j not in desired path.
        for j in range(len(edges)):
            if xzero[j] == 0:
                constraints.append(lambda_[j] >= 0)
    # some locations are forced to be free (cant add obstacles)
    for i in range(len(varnodes)):
        if varnodes[i] in forced_free_locations:
            idx = len(allowed_area_types) * i
            constraints.append(l_[idx+0] == 1)
    # Cost
    cost = cp.norm1(l_ - l_original)
    # solve with cvxpy
    prob = cp.Problem(cp.Minimize(cost), constraints)
    value = prob.solve(solver=cp.GUROBI)
    if value == float('inf'):
        print("inverse shortest path FAILED")
        return [], False, []

    # new graph - weights added to edges
    new_graph = graph.copy()
    l_new = [round(i) for i in l_.value]
    for i in range(len(l_original)):
        if l_original[i] != l_new[i]:
            vn_idx = i // len(allowed_area_types)
            vn = varnodes[vn_idx]
            new_graph.nodes[vn]['area_type'] = l_new[i]
    for vn in varnodes:
        area_type = new_graph.nodes[vn]['area_type']
        for adj in new_graph[vn]:
            new_graph[adj][vn]['weight'] = area_costs[area_type] * dist_per_edge

    # New obstacles set
    new_obstacles = []
    #for i in range(len(l_original)):
    #    if l_original[i] != l_new[i]:
    #        vn_idx = i // len(allowed_area_types)
    #        vn = varnodes[vn_idx]
    #        new_obstacles.append(list(vn))
    for i in range(len(varnodes)):
        idx = len(allowed_area_types) * i
        if round(l_new[idx+1]) == 1:
            vn = varnodes[i]
            new_obstacles.append(list(vn))

    # Sanity Check
    success = check(dp, new_graph)

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
                    l_new = [round(i) for i in sol]
                    sol_obstacles = []
                    for i in range(len(varnodes)):
                        idx = len(allowed_area_types) * i
                        if round(l_new[idx+1]) == 1:
                            vn = varnodes[i]
                            sol_obstacles.append(list(vn))
                    all_solution_obstacles.append(sol_obstacles)
                return new_graph, True, all_solution_obstacles
            all_solutions.append(l_.value)

    # Return
    return new_graph, success, new_obstacles


def main_inv_mapf(problem_file, verbose=False, animate=False, question_partial_plan=False, find_all_solutions=False):
    problem_fullpath = EXAMPLES_PATH + "/" + problem_file
    return main_inv_mapf_fullpath(problem_fullpath, verbose, animate, question_partial_plan, find_all_solutions)


def main_inv_mapf_fullpath(problem_fullpath, verbose=False, animate=False, question_partial_plan=False, find_all_solutions=False):

    # Parsing and generating CBS solution of original MULTI-AGENT problem
    solved = explanations_multi.generate_cbs_solution(problem_fullpath)
    if not solved:
        return False, []
    raw_problem = explanations_multi.parse_yaml(problem_fullpath)
    raw_solution = explanations_multi.parse_yaml(SOLUTION_YAML)

    # Handling desired path of the agent and get agent name
    desired_paths = []
    agent_names = []
    for agent in raw_problem['agents']:
        if agent.get('waypoints') is not None:
            desired_paths.append(agent['waypoints'])
            agent_names.append(agent['name'])
    #if len(desired_paths) > 1:
    #  print('For now, only supporting this baseline if there is a desired path for a single agent only')
    #  return False, []

    # Get paths (schedule) of all agents except those that have a desired path
    ori_makespan = copy.deepcopy(raw_solution['statistics']['makespan'])
    schedule = copy.deepcopy(raw_solution['schedule'])
    for agent_name in agent_names:
        schedule.pop(agent_name)

    # Get set of all the nodes that are ever passed by any agent (other than ours) at any time
    nodes_passed = []
    for path in raw_solution['schedule'].values():
        nodes_passed.append( (path[-1]['x'], path[-1]['y']) )
    if not question_partial_plan:
        for path in schedule.values():
            for t, pos in enumerate(path):
                n = (pos['x'], pos['y'])
                nodes_passed.append(n)

    # Make cells from desired paths also forcefully free
    for i in range(len(desired_paths)):
        for x in desired_paths[i]:
            nodes_passed.append( (x[0],x[1]) )

    # Get graph for single-agent ISP
    graph, old_dct = parse_yaml(problem_fullpath)

    # Solve joint-ISP
    new_graph, success_isp, all_obstacle_solutions = isp_discrete(graph, agent_names, desired_paths, forced_free_locations=nodes_passed, find_all_solutions=find_all_solutions)
    if not success_isp:
      return False, []

    if find_all_solutions:

        # all (optimal) solutions
        all_solutions = []

        for new_obstacles in all_obstacle_solutions:

            # Create new schedule and problem dict and created a new problem yaml file
            new_schedule = explanations_multi.create_new_schedule(raw_solution, desired_paths, agent_names)
            new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, new_obstacles)
            new_filename = "additional/build/new_problem.yaml"
            explanations_multi.create_problem_yaml(new_problem, new_filename)

            # Sanity Check
            solved = explanations_multi.generate_cbs_solution(ROOT_PATH + "/" + new_filename)
            if not solved:
                continue
            new_cbs_solution = explanations_multi.parse_yaml(SOLUTION_YAML)
            if question_partial_plan:
                success = explanations_multi.does_mapf_solution_satisfy_desired_paths(new_problem, new_cbs_solution, agent_names, desired_paths)
            else:
                success = explanations_multi.does_mapf_solution_satisfy_desired_paths_with_other_agents_fixed(new_problem, raw_solution, new_cbs_solution, agent_names, desired_paths)
            if success:
                all_solutions.append(new_obstacles)

        new_obstacles = all_obstacle_solutions[0]

    else:

        # single (optimal) solution
        new_obstacles = all_obstacle_solutions

        if len(new_obstacles) == 0:
          print('No obstacles added/removed! Means single-agent approach cannot solve this multi-agent ISP problem')
          return False, []

        # Create new schedule and problem dict and created a new problem yaml file
        new_schedule = explanations_multi.create_new_schedule(raw_solution, desired_paths, agent_names)
        new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, new_obstacles)
        new_filename = "additional/build/new_problem.yaml"
        explanations_multi.create_problem_yaml(new_problem, new_filename)

        # Sanity Check
        solved = explanations_multi.generate_cbs_solution(ROOT_PATH + "/" + new_filename)
        if not solved:
            return False, []
        new_cbs_solution = explanations_multi.parse_yaml(SOLUTION_YAML)
        if question_partial_plan:
            success = explanations_multi.does_mapf_solution_satisfy_desired_paths(new_problem, new_cbs_solution, agent_names, desired_paths)
        else:
            success = explanations_multi.does_mapf_solution_satisfy_desired_paths_with_other_agents_fixed(new_problem, raw_solution, new_cbs_solution, agent_names, desired_paths)

    # Animation
    if animate and success and len(new_obstacles) > 0:
        print('Showing original plan on new map')
        explanations_multi.generate_animation(raw_problem, new_problem, raw_solution)

        print('Showing desired plan on new map')
        desired_mapf_solution = copy.deepcopy(raw_solution)
        for i in range(len(desired_paths)):
            agent = agent_names[i]
            desired_mapf_solution['schedule'][agent] = []
            for t, pos in enumerate(desired_paths[i]):
                desired_mapf_solution['schedule'][agent].append( {'x':pos[0], 'y':pos[1], 't':t} )
        explanations_multi.generate_animation(raw_problem, new_problem, desired_mapf_solution)

        print('Showing new solution for new map')
        explanations_multi.generate_animation(raw_problem, new_problem, new_cbs_solution)

    # return
    if find_all_solutions:
        return len(all_solutions) > 0, all_obstacle_solutions
    else:
        return success, new_obstacles


if __name__ == '__main__':

    # Terminal Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_file", help="input problem filepath")
    parser.add_argument("-v", "--verbose", action="store_true", help="outputs debug information")
    parser.add_argument("-a", "--animate", action="store_true", help="shows animation")
    parser.add_argument("-q", "--question-partial-plan", action="store_true", help="solves question for partial (instead of full) plan, i.e. 'why do agents A not take paths D?' instead of 'why not full plan X?'")
    args = parser.parse_args()
    main_inv_mapf(args.problem_file, args.verbose, args.animate, args.question_partial_plan)
