import os
import glob
import random
import yaml
import networkx as nx
import copy
import pdb
import explanations_multi
import explanations_multi_incremental
import explanations_global
import baseline_single_agent
from path import *
#from memory_profiler import profile

ANIMATE = False
METHOD = 'multi' # 'single', 'multi', 'incr'

def get_simple_graph(filepath, path_unpickable):
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
            # skip obstacles
            if (r,c) in path_unpickable:
              continue
            # horizontal edges
            if c < dimensions[0] - 1:
                if (r, c+1) not in path_unpickable:
                  graph.add_edge((r, c), (r, c + 1))
                  graph.add_edge((r, c + 1), (r, c))
            # vertical edges
            if r < dimensions[1] - 1:
                if (r+1, c) not in path_unpickable:
                  graph.add_edge((r, c), (r + 1, c))
                  graph.add_edge((r + 1, c), (r, c))
    # Return
    return graph

def my_create_new_problem(old_problem, new_paths, agent_names):
    new_problem = copy.deepcopy(old_problem)
    for agent in new_problem['agents']:
        for i, agent_name in enumerate(agent_names):
            if agent['name'] == agent_name:
                agent['start'] = list(new_paths[i][0])
                agent['goal'] = list(new_paths[i][-1])
                agent['waypoints'] = [ [c[0], c[1]] for c in new_paths[i] ]
    return new_problem

#@profile
def generate_problem(f, raw_problem, raw_solution):
    # pick a random agent
    found = False
    for i in range(100):
        agent = random.choice(raw_problem['agents'])
        if agent['start'] != agent['goal']:
            found = True
            break
    if not found:
        print('Could not find a good agent... giving up.')
        return False

    # unpickable waypoints are obstacles, start/goal of all agents, current path of our agent
    unpickable = []
    for obs in raw_problem['map']['obstacles']:
        unpickable.append( tuple(obs) )
    for path in raw_solution['schedule'].values():
        unpickable.append( (path[0]['x'], path[0]['y']) )
        unpickable.append( (path[-1]['x'], path[-1]['y']) )
    for t, pos in enumerate(raw_solution['schedule'][agent['name']]):
          unpickable.append( (pos['x'], pos['y']) )

    # pick a random waypoint
    found = False
    for i in range(10000):
        waypoint = (random.randint(0, raw_problem['map']['dimensions'][0]-1), random.randint(0, raw_problem['map']['dimensions'][1]-1))
        if waypoint not in unpickable:
            found = True
            break
    if not found:
        print('Could not find a good waypoint... giving up.')
        return False

    # obstacles and other agents' goals
    path_unpickable = []
    for obs in raw_problem['map']['obstacles']:
        path_unpickable.append( tuple(obs) )
    for a in raw_solution['schedule']:
        if a != agent['name']:
            path = raw_solution['schedule'][a]
            path_unpickable.append( (path[-1]['x'], path[-1]['y']) )

    # compute shortest path through waypoint, that avoids obstacles and other agents' goals
    graph = get_simple_graph(f, path_unpickable)
    try:
        p1 = nx.shortest_path(graph, source=tuple(agent['start']), target=waypoint, weight="weight")
        p2 = nx.shortest_path(graph, source=waypoint, target=tuple(agent['goal']), weight="weight")
        desired_path = p1 + p2[1:]
    except nx.exception.NetworkXNoPath:
        desired_path = None
    except nx.exception.NodeNotFound:
        desired_path = None

    if desired_path == None:
        return False
    if len(desired_path) == len(raw_solution['schedule'][agent['name']]):
        return False

    # Check for collisions
    for a in raw_solution['schedule']:
        if a != agent['name']:
            path = raw_solution['schedule'][a]
            for t, pos in enumerate(path):
                n = (pos['x'], pos['y'])
                if n == tuple(desired_path[min(t, len(desired_path) - 1)]):
                    print('Collision between desired path and other agents')
                    return False
    for t in range(raw_solution['statistics']['makespan'], len(desired_path)):
        for a in raw_solution['schedule']:
            if a != agent['name']:
                path = raw_solution['schedule'][a]
                last_node = (path[-1]['x'], path[-1]['y'])
                if tuple(desired_path[t]) == last_node:
                    print('Collision between desired path and other agents')
                    return False

    print('Desired path for %s has new length %d (it was %d before)' % (agent['name'], len(desired_path), len(raw_solution['schedule'][agent['name']])))

    # save this as an InvMAPF problem...
    new_problem = my_create_new_problem(raw_problem, [desired_path], [agent['name']])
    new_filename = "rnd_problem.yaml"
    explanations_multi.create_problem_yaml(new_problem, new_filename)

    # try to solve it
    if METHOD == 'multi':
        try:
            success, new_obstacles = explanations_multi.main_inv_mapf('../'+new_filename, False, ANIMATE)
        except KeyError:
            success = False
    elif METHOD == 'incr':
        try:
            success, new_obstacles = explanations_multi_incremental.main_inv_mapf('../'+new_filename, False, ANIMATE)
        except KeyError:
            success = False
    elif METHOD == 'global':
        try:
            success, new_obstacles = explanations_global.main_inv_mapf('../'+new_filename, False, ANIMATE)
        except KeyError:
            success = False
    elif METHOD == 'single':
        try:
            success, new_obstacles = baseline_single_agent.main_inv_mapf('../'+new_filename, False, ANIMATE)
        except KeyError:
            success = False
    else:
        print('No such method')
        exit()

    if len(new_obstacles) == len(raw_problem['map']['obstacles']):
        success = False

    # save the problem if it is solvable
    if success:
        print('FOUND A GOOD INV-MAPF PROBLEM!!!')
        
        new_filename_save = 'rnd_' + METHOD + '_inv_problem_' + os.path.basename(f)
        explanations_multi.create_problem_yaml(new_problem, new_filename_save)
        return True
    else:
        return False


# Main
if __name__ == '__main__':

    files = sorted(glob.glob(ROOT_PATH + '/../../examples/agents5/*.yaml'))

    # see what we have generated so far
    start_idx = 0
    for i in range(len(files)):
        f = files[i]
        new_filename_save = 'rnd_' + METHOD + '_inv_problem_' + os.path.basename(f)
        if os.path.exists(new_filename_save):
            start_idx = i+1

    # generate one inv-mapf problem for each raw problem
    for i in range(start_idx, len(files)):

        f = files[i]
        print(f)

        explanations_multi.generate_cbs_solution(f)
        raw_problem = explanations_multi.parse_yaml(f)
        raw_solution = explanations_multi.parse_yaml(SOLUTION_YAML)

        for invmapf_attempt in range(100):
            if generate_problem(f, raw_problem, raw_solution):
                break

