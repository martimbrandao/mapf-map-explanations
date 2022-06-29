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


def create_aux_vars(graph, desired_path):
    # Nodes, edges, non-portal nodes (varnodes), insert portals to desired path
    dp = insert_portals(desired_path)
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

    # Desired x
    xzero = np.zeros(len(edges))
    for p in range(len(dp) - 1):
        j = edge2index[dp[p], dp[p + 1]]
        xzero[j] = 1

    # Return
    return dp, nodes, edges, edge2index, varnodes, edge2varnodeindex, A, xzero


# Sanity Check
def check(dp, new_graph):
    success = False
    new_path = nx.shortest_path(new_graph, source=dp[0], target=dp[-1], weight="weight")
    if get_cost(new_graph, new_path) == get_cost(new_graph, dp):
        success = True
        print("desired path is optimal: yes")
    else:
        print("desired path is optimal: no")
    return success


# ISP Discrete
def isp_discrete(graph, desired_path, area_costs=None, allowed_area_types=None, dist_per_edge=0.5, forced_free_locations=[], bad_solutions=[]):
    if allowed_area_types is None:
        allowed_area_types = [0, 1]
    if area_costs is None:
        area_costs = [1, 1000]

    dp, nodes, edges, edge2index, varnodes, edge2varnodeindex, A, xzero = create_aux_vars(graph, desired_path)

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
    # Variables
    l_ = cp.Variable(len(l_original), boolean=True)
    pi_ = cp.Variable(len(nodes))
    lambda_ = cp.Variable(len(edges))
    # Cost
    cost = cp.norm1(l_ - l_original)
    # Constraints
    constraints = []
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
    # NOTE THIS IS DIFFERENT FROM ORIGINAL CONSTRAINTS (ORIGINAL: >= 0)
    # Otherwise new obstacles are not created
    # probably justified because this is a 'sensitivity' parameter
    for j in range(len(edges)):
        if xzero[j] == 0:
            constraints.append(lambda_[j] >= 0)
    # some locations are forced to be free (cant add obstacles)
    for i in range(len(varnodes)):
        if varnodes[i] in forced_free_locations:
            idx = len(allowed_area_types) * i
            constraints.append(l_[idx+0] == 1)
    # some solutions are forbidden (we know they won't be feasible)
    for bad_l in bad_solutions:
        diff = (bad_l==1) * 1 - (bad_l==0) * 1
        constraints.append( diff @ l_ <= np.sum(bad_l==1) - 1 )
    # solve with cvxpy
    prob = cp.Problem(cp.Minimize(cost), constraints)
    value = prob.solve(solver=cp.GUROBI)
    if value == float('inf'):
        print("inverse shortest path FAILED")
        return [], False, [], []

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
    # Return
    return new_graph, success, new_obstacles, l_.value


def isp_discrete_incremental(graph, desired_path, initial_bad_path, area_costs=None, allowed_area_types=None, dist_per_edge=0.5, forced_free_locations=[], bad_solutions=[]):

    dp = insert_portals(desired_path)
    bad_paths = [insert_portals(initial_bad_path)]
    prev_graph = []
    prev_success = False
    prev_obstacles = []
    prev_vars = []
    while True:
        new_graph, success, new_obstacles, new_vars = isp_discrete_incremental_step(graph, desired_path, bad_paths, area_costs, allowed_area_types, dist_per_edge, forced_free_locations, bad_solutions)
        # if I could not solve the optimization problem
        if new_graph == []:
            return prev_graph, prev_success, prev_obstacles, prev_vars
        # if the dedired path is optimal
        if success:
            return new_graph, success, new_obstacles, new_vars
        # otherwise the desired path is not optimal but we have made changes to the map, so add the current optimal path to the list of "bad_paths"
        new_path = nx.shortest_path(new_graph, source=dp[0], target=dp[-1], weight="weight")
        bad_paths.append(new_path)
        # update
        prev_graph = new_graph
        prev_success = success
        prev_obstacles = new_obstacles
        prev_vars = new_vars

def isp_discrete_incremental_step(graph, desired_path, bad_paths, area_costs=None, allowed_area_types=None, dist_per_edge=0.5, forced_free_locations=[], bad_solutions=[]):
    if allowed_area_types is None:
        allowed_area_types = [0, 1]
    if area_costs is None:
        area_costs = [1, 1000]

    dp, nodes, edges, edge2index, varnodes, edge2varnodeindex, A, xzero = create_aux_vars(graph, desired_path)

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
    # Variables
    l_ = cp.Variable(len(l_original), boolean=True)

    # Cost
    cost = cp.norm1(l_ - l_original)

    # Constraints
    constraints = []

    # Constraints [G @ l_ <= h]
    #     sum_(j in good_path) 1000 * l_j + 1 <= sum_(j in bad_path) 1000 * l_j + 1
    # so: sum_(j in good_path) 1000 * l_j - sum_(j in bad_path) 1000 * l_j <= sum_(j in bad_path) 1 - sum_(j in good_path) 1
    G = []
    h = []
    for bad_path in bad_paths:
        bp = insert_portals(bad_path)
        Gline = [0.0] * len(l_original)
        hline = 0
        for i in range(len(varnodes)):
            if varnodes[i] in dp:
                j = len(allowed_area_types) * i
                Gline[j+0] += 1
                Gline[j+1] += 1000
                #hline -= 1
        for i in range(len(varnodes)):
            if varnodes[i] in bp:
                j = len(allowed_area_types) * i
                Gline[j+0] -= 1
                Gline[j+1] -= 1000
                #hline += 1
        G.append(Gline)
        h.append(hline)
    if len(G) > 0:
        G = np.array(G)
        h = np.array(h)
        constraints.append(G @ l_ <= h)

    # sum_k l_ik = 1, for all i
    for i in range(len(varnodes)):
        idx = len(allowed_area_types) * i
        constraints.append(cp.sum(l_[idx:idx + len(allowed_area_types)]) == 1)

    # some locations are forced to be free (cant add obstacles)
    for i in range(len(varnodes)):
        if varnodes[i] in forced_free_locations:
            idx = len(allowed_area_types) * i
            constraints.append(l_[idx+0] == 1)

    # some solutions are forbidden (we know they won't be feasible)
    for bad_l in bad_solutions:
        diff = (bad_l==1) * 1 - (bad_l==0) * 1
        constraints.append( diff @ l_ <= np.sum(bad_l==1) - 1 )

    # solve with cvxpy
    prob = cp.Problem(cp.Minimize(cost), constraints)
    value = prob.solve(solver=cp.GUROBI)
    if value == float('inf'):
        print("inverse shortest path FAILED")
        return [], False, [], []

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
    # Return
    return new_graph, success, new_obstacles, l_.value


def main_inv_mapf(problem_file, verbose=False, animate=False):
    problem_fullpath = EXAMPLES_PATH + "/" + problem_file
    return main_inv_mapf_fullpath(problem_fullpath, verbose, animate)


def main_inv_mapf_fullpath(problem_fullpath, verbose=False, animate=False):

    dist_per_edge = 0.5
    area_costs = [1, 1000]

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
    if len(desired_paths) > 1:
      print('For now, only supporting this baseline if there is a desired path for a single agent only')
      exit()
    desired_path = [(x[0], x[1]) for x in desired_paths[0]]

    initial_bad_path = [(pos['x'], pos['y']) for pos in raw_solution['schedule'][agent_names[0]]]

    # Get paths (schedule) of all agents except those that have a desired path
    ori_makespan = copy.deepcopy(raw_solution['statistics']['makespan'])
    schedule = copy.deepcopy(raw_solution['schedule'])
    for agent_name in agent_names:
        schedule.pop(agent_name)

    # Get cells where we can't place obstacles (starts and goals of all agents)
    forced_free_locations = []
    for path in raw_solution['schedule'].values():
        forced_free_locations.append( (path[0]['x'], path[0]['y']) )
        forced_free_locations.append( (path[-1]['x'], path[-1]['y']) )
    for x in desired_path:
        forced_free_locations.append(x)

    # Get graph for single-agent ISP
    graph, old_dct = parse_yaml(problem_fullpath)
    dp, _, _, _, varnodes, _, _, _ = create_aux_vars(graph, desired_path)
    for vn in varnodes:
        area_type = graph.nodes[vn]['area_type']
        for adj in graph[vn]:
            graph[adj][vn]['weight'] = area_costs[area_type] * dist_per_edge

    # if desired paths already optimal
    if check(dp, graph):
        return True, raw_problem['map']['obstacles']

    # loop
    bad_solutions = []
    success = False
    while not success:

        print('--- Solving InvSP with %d forbidden solutions...' % len(bad_solutions))
        #print(str(bad_solutions))

        # Run single-agent ISP with the extra constraint that we cannot add obstacles to places that other agents currently pass
        new_graph, success_isp, new_obstacles, new_vars = isp_discrete_incremental(graph, desired_path, initial_bad_path, forced_free_locations=forced_free_locations, bad_solutions=bad_solutions)
        if new_graph == []:
          return False, []

        # Create new problem dict and create a new problem yaml file
        new_problem = explanations_multi.create_new_problem(raw_problem, desired_paths, agent_names, new_obstacles)
        new_filename = "additional/build/new_problem.yaml"
        explanations_multi.create_problem_yaml(new_problem, new_filename)

        # Check if desired paths are optimal
        solved = explanations_multi.generate_cbs_solution(ROOT_PATH + "/" + new_filename)
        if not solved:
            return False, []
        new_cbs_solution = explanations_multi.parse_yaml(SOLUTION_YAML)
        success = explanations_multi.is_mapf_solution_valid(new_problem, new_cbs_solution, agent_names, desired_paths)
        if not success:
            bad_solutions.append(new_vars)
        #if animate:
        #    explanations_multi.generate_animation(raw_problem, new_problem, new_cbs_solution)
        #    #desired_solution = copy.deepcopy(raw_solution)
        #    #desired_solution['schedule'][agent_names[0]] = []
        #    #for t, pos in enumerate(desired_paths[0]):
        #    #    desired_solution['schedule'][agent_names[0]].append( {'x':pos[0], 'y':pos[1], 't':t} )
        #    #explanations_multi.generate_animation(raw_problem, new_problem, desired_solution)

    print('Found feasible InvMAPF solution!')

    if animate:
        explanations_multi.generate_animation(raw_problem, new_problem, raw_solution)
        explanations_multi.generate_animation(raw_problem, new_problem, new_cbs_solution)

    return success, new_obstacles

if __name__ == '__main__':

    # Terminal Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_file", help="input problem filepath")
    parser.add_argument("-v", "--verbose", action="store_true", help="outputs debug information")
    parser.add_argument("-a", "--animate", action="store_true", help="shows animation")
    args = parser.parse_args()
    main_inv_mapf(args.problem_file, args.verbose, args.animate)
