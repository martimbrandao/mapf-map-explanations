import os
import glob
import random
import yaml
import networkx as nx
import numpy as np
import copy
import pdb
import subprocess
import resource
import explanations_multi
import explanations_multi_incremental
import explanations_global
import baseline_single_agent
import baseline_joint_single_agent
import problem_generator
from path import *
#from memory_profiler import profile

MAX_AGENTS = 30

ANIMATE = False
METHOD = 'incr' # 'single', 'jointsingle', 'multi', 'incr', 'global'
QUESTION_PARTIAL_PLAN = True
PROBLEM_GENERATION_MODE = 'holes' # 'holes', 'center'

MAX_TIME = 60  # Maximum search time (in seconds)...
MAX_VIRTUAL_MEMORY = 8 * 1024 * 1024 * 1024 # Maximal virtual memory for subprocesses (in bytes)...


def load_scen(map_file, scen_file, max_agents):

  print('')

  # load .map
  print(map_file)
  with open(map_file) as f:
      lines = f.readlines()
  lines.pop(0)
  lines.pop(0)
  lines.pop(0)
  lines.pop(0)
  mymap = np.array(lines, dtype=('U1',len(lines[0])))
  mymap = mymap[:,:-1]

  obstacles = []
  for i in range(mymap.shape[0]):
    for j in range(mymap.shape[1]):
      if mymap[i,j] == 'T':
        obstacles.append([i,j])
  mymap2 = {'dimensions': [mymap.shape[0], mymap.shape[1]], 'obstacles': obstacles}

  # load .scen
  print(scen_file)
  with open(scen_file) as f:
      lines = f.readlines()
  lines.pop(0)

  agents = []
  for l in range(min(max_agents, len(lines))):
    ag = lines[l].split('\t')
    agents.append({'start': [int(ag[5]), int(ag[4])], 'goal': [int(ag[7]), int(ag[6])], 'name': 'agent'+str(l)})

  # yaml problem
  raw_problem = {'agents': agents, 'map': mymap2}
  return raw_problem, mymap


def get_simple_graph(problem, path_unpickable):
    dimensions = problem['map']['dimensions']
    obstacles = problem['map']['obstacles']
    # Create Graph
    graph = nx.Graph()
    for r in range(dimensions[0]):
        for c in range(dimensions[1]):
            # skip obstacles
            if (r,c) in path_unpickable:
              continue
            # horizontal edges
            if c < dimensions[1] - 1:
                if (r, c+1) not in path_unpickable:
                  graph.add_edge((r, c), (r, c + 1), weight=1)
                  graph.add_edge((r, c + 1), (r, c), weight=1)
            # vertical edges
            if r < dimensions[0] - 1:
                if (r+1, c) not in path_unpickable:
                  graph.add_edge((r, c), (r + 1, c), weight=1)
                  graph.add_edge((r + 1, c), (r, c), weight=1)
    # Return
    return graph


def generate_cbsh2_solution(mapfile, scenfile, num_agents):
    os.chdir(CBS_DIR_PATH)

    # solve
    if os.path.exists('paths.txt'):
        os.remove('paths.txt')
    cmd = './cbsh2 -m ' + mapfile + ' -a ' + scenfile + ' -o test.csv --outputPaths=paths.txt -k ' + str(num_agents) + ' -t ' + str(MAX_TIME)
    out1, err1 = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    success = os.path.exists('paths.txt')

    # get paths
    solution = {}
    if success:
      with open('paths.txt') as f:
        lines = f.readlines()
      solution['schedule'] = {}
      for line in lines:
        i_name = line.find(':')
        agent_name = 'agent' + line[6:i_name]
        solution['schedule'][agent_name] = []
        path = line[i_name+2:].split('->')
        for t in range(len(path)-1):
          pos = path[t]
          i_comma = pos.find(',')
          i = int(pos[1:i_comma])
          j = int(pos[i_comma+1:-1])
          solution['schedule'][agent_name].append({'x': i, 'y': j, 't': t})

    os.chdir(ROOT_PATH)
    return success, solution


# Main
if __name__ == '__main__':

    folder = ROOT_PATH + '/warehouse-10-20-10-2-1'
    map_file = folder+'/warehouse-10-20-10-2-1.map'
    scen_files  = sorted(glob.glob(folder+'/scen-even/*.scen'))
    scen_files += sorted(glob.glob(folder+'/scen-random/*.scen'))

    #for scen_file in ['/home/k1929632/workspace/martim/xaip/invmapf/multi-agent/inv-shortest-path/warehouse-10-20-10-2-1/scen-even/warehouse-10-20-10-2-1-even-24.scen']: 
    for scen_file in scen_files:

      basename = os.path.splitext(os.path.basename(scen_file))[0]
      raw_problem, raw_map = load_scen(map_file, scen_file, MAX_AGENTS)

      ########################################################################
      # try invMAPF problem through center of map
      if PROBLEM_GENERATION_MODE == 'center':

          generated = False
          for agent in raw_problem['agents']:
              if generated:
                  break
              a = agent['name']

              # obstacles and other agents' goals
              path_unpickable = [tuple(obs) for obs in raw_problem['map']['obstacles']]
              for aaa in raw_problem['agents']:
                  if aaa['name'] != a:
                      path_unpickable.append( tuple(aaa['goal']) )

              # we want this agent to take its shortest path (ignoring other agents)
              graph = get_simple_graph(raw_problem, path_unpickable)
              try:
                  #desired_path = nx.shortest_path(graph, source=tuple(agent['start']), target=tuple(agent['goal']), weight="weight")
                  waypoint = (int(raw_map.shape[0]/2), int(raw_map.shape[1]/2))
                  p1 = nx.shortest_path(graph, source=tuple(agent['start']), target=waypoint, weight="weight")
                  p2 = nx.shortest_path(graph, source=waypoint, target=tuple(agent['goal']), weight="weight")
                  desired_path = p1 + p2[1:]
              except nx.exception.NetworkXNoPath:
                  desired_path = None
              except nx.exception.NodeNotFound:
                  desired_path = None
              if desired_path == None:
                  print('No path through waypoint')
                  continue

              # save this as an InvMAPF problem...
              new_problem = problem_generator.my_create_new_problem(raw_problem, [desired_path], [a])
              new_filename = 'rnd_problem.yaml'
              explanations_multi.create_problem_yaml(new_problem, new_filename)

              # try to solve it
              new_obstacles = raw_problem['map']['obstacles']
              if METHOD == 'multi':
                  if QUESTION_PARTIAL_PLAN == True:
                      print('Cannot answer partial-plan questions with this method.')
                      exit()
                  try:
                      success, new_obstacles = explanations_multi.main_inv_mapf('../'+new_filename, False, ANIMATE)
                  except KeyError:
                      success = False
              elif METHOD == 'incr':
                  try:
                      success, new_obstacles = explanations_multi_incremental.main_inv_mapf('../'+new_filename, False, ANIMATE, QUESTION_PARTIAL_PLAN)
                  except KeyError:
                      success = False
              elif METHOD == 'global':
                  if QUESTION_PARTIAL_PLAN == False:
                      print('Cannot answer full-plan questions with this method.')
                      exit()
                  try:
                      success, new_obstacles = explanations_global.main_inv_mapf('../'+new_filename, False, ANIMATE)
                  except KeyError:
                      success = False
              elif METHOD == 'single':
                  try:
                      success, new_obstacles = baseline_single_agent.main_inv_mapf('../'+new_filename, False, ANIMATE, QUESTION_PARTIAL_PLAN)
                  except KeyError:
                      success = False
              elif METHOD == 'jointsingle':
                  try:
                      success, new_obstacles = baseline_joint_single_agent.main_inv_mapf('../'+new_filename, False, ANIMATE, QUESTION_PARTIAL_PLAN)
                  except KeyError:
                      success = False
              else:
                  print('No such method')
                  exit()

              if len(new_obstacles) == len(raw_problem['map']['obstacles']) or len(new_obstacles) == 0:
                  success = False

              # save the problem if it is solvable
              if success:
                  print('FOUND A GOOD INV-MAPF PROBLEM!!!')
                  new_filename_save = 'rnd1_' + METHOD + '_inv_problem_' + basename + '.yaml'
                  explanations_multi.create_problem_yaml(new_problem, new_filename_save)
                  generated = True
                  break


      ########################################################################
      # try invMAPF problem by making holes in shelves, then seeing if an agent uses it, asking why it doesnt take its shortest path
      if PROBLEM_GENERATION_MODE == 'holes':

          # identify shelf-obstacles
          shelf_obstacles = []
          for obs in raw_problem['map']['obstacles']:
              if obs[0] > 0 and obs[0] < raw_problem['map']['dimensions'][0]-1 and obs[1] > 0 and obs[1] < raw_problem['map']['dimensions'][1]-1:
                  # count how many neighboring obstacles
                  nei_obs = 0
                  nei_added = 0
                  for a in [-1,-0,1]:
                      for b in [-1,0,1]:
                          if [obs[0]+a, obs[1]+b] in raw_problem['map']['obstacles']:
                              nei_obs += 1
                          if [obs[0]+a, obs[1]+b] in shelf_obstacles:
                              nei_added += 1
                  # if this is not a corner (nei_obs > 4)
                  if nei_obs > 4 and nei_added == 0:
                      shelf_obstacles.append(obs)

          # several attempts
          attempts = 0
          generated = False
          while generated == False and attempts < 1000:

              attempts += 1

              # remove some shelf-obstacles
              obstacles_to_remove = random.choices(shelf_obstacles, k=20)
              raw_obstacles_plus_holes = []
              for obs in raw_problem['map']['obstacles']:
                  if obs not in obstacles_to_remove:
                      raw_obstacles_plus_holes.append(obs)
              raw_problem_plus_holes = copy.deepcopy(raw_problem)
              raw_problem_plus_holes['map']['obstacles'] = raw_obstacles_plus_holes

              # generate new map file
              raw_map_plus_holes = copy.deepcopy(raw_map)
              for obs in obstacles_to_remove:
                  raw_map_plus_holes[obs[0],obs[1]] = '.'
              file_map_plus_holes = ROOT_PATH + '/rnd_map.map'
              with open(file_map_plus_holes, 'w') as f:
                  f.write('type octile\n')
                  f.write('height ' + str(raw_map.shape[0]) + '\n')
                  f.write('width ' + str(raw_map.shape[1]) + '\n')
                  f.write('map\n')
                  for i in range(raw_map.shape[0]):
                      for j in range(raw_map.shape[1]):
                          f.write(raw_map_plus_holes[i,j])
                      f.write('\n')

              # solve quickly with cbsh2
              solved, new_solution = generate_cbsh2_solution(file_map_plus_holes, scen_file, MAX_AGENTS)
              if not solved:
                  print('not solved')
                  continue

              # check if solution uses empty space created by obstacle removal
              for agent in raw_problem_plus_holes['agents']:
                  if generated:
                      break
                  a = agent['name']
                  for pos in new_solution['schedule'][a]:
                      if [pos['x'],pos['y']] in obstacles_to_remove:

                          # found an agent using shelf holes
                          print('found an agent using shelf holes!!')
                          if ANIMATE:
                              explanations_multi.generate_animation(raw_problem, raw_problem_plus_holes, new_solution)

                          # obstacles and other agents' goals
                          path_unpickable = [tuple(obs) for obs in raw_problem['map']['obstacles']]
                          for aaa in raw_problem['agents']:
                              if aaa['name'] != a:
                                  path_unpickable.append( tuple(aaa['goal']) )

                          # we want this agent to take its shortest path (ignoring holes and other agents)
                          graph = get_simple_graph(raw_problem, path_unpickable)
                          try:
                              desired_path = nx.shortest_path(graph, source=tuple(agent['start']), target=tuple(agent['goal']), weight="weight")
                          except nx.exception.NetworkXNoPath:
                              desired_path = None
                          except nx.exception.NodeNotFound:
                              desired_path = None
                          if desired_path == None:
                              print('No path through waypoint')
                              continue

                          #
                          if len(desired_path) == len(new_solution['schedule'][a]):
                              print('whether we traverse the shelf hole or not, it does not change path length')
                              #explanations_multi.generate_animation(raw_problem, raw_problem_plus_holes, new_solution)
                              #myprob = copy.deepcopy(raw_problem)
                              #myprob2 = copy.deepcopy(raw_problem_plus_holes)
                              #myprob['agents'] = [agent]
                              #myprob2['agents'] = [agent]
                              #mysol2 = copy.deepcopy(new_solution)
                              #mysol2['schedule'] = {a: new_solution['schedule'][a]}
                              #explanations_multi.generate_animation(myprob, myprob2, mysol2)
                              continue

                          # save this as an InvMAPF problem...
                          new_problem = problem_generator.my_create_new_problem(raw_problem_plus_holes, [desired_path], [a])
                          new_filename = 'rnd_problem.yaml'
                          explanations_multi.create_problem_yaml(new_problem, new_filename)

                          # try to solve it
                          new_obstacles = raw_problem_plus_holes['map']['obstacles']
                          if METHOD == 'multi':
                              if QUESTION_PARTIAL_PLAN == True:
                                  print('Cannot answer partial-plan questions with this method.')
                                  exit()
                              try:
                                  success, new_obstacles = explanations_multi.main_inv_mapf('../'+new_filename, False, ANIMATE)
                              except KeyError:
                                  success = False
                          elif METHOD == 'incr':
                              try:
                                  success, new_obstacles = explanations_multi_incremental.main_inv_mapf('../'+new_filename, False, ANIMATE, QUESTION_PARTIAL_PLAN)
                              except KeyError:
                                  success = False
                          elif METHOD == 'global':
                              if QUESTION_PARTIAL_PLAN == False:
                                  print('Cannot answer full-plan questions with this method.')
                                  exit()
                              try:
                                  success, new_obstacles = explanations_global.main_inv_mapf('../'+new_filename, False, ANIMATE)
                              except KeyError:
                                  success = False
                          elif METHOD == 'single':
                              try:
                                  success, new_obstacles = baseline_single_agent.main_inv_mapf('../'+new_filename, False, ANIMATE, QUESTION_PARTIAL_PLAN)
                              except KeyError:
                                  success = False
                          elif METHOD == 'jointsingle':
                              try:
                                  success, new_obstacles = baseline_joint_single_agent.main_inv_mapf('../'+new_filename, False, ANIMATE, QUESTION_PARTIAL_PLAN)
                              except KeyError:
                                  success = False
                          else:
                              print('No such method')
                              exit()

                          if len(new_obstacles) == len(raw_problem_plus_holes['map']['obstacles']):
                              success = False

                          # save the problem if it is solvable
                          if success:
                              print('FOUND A GOOD INV-MAPF PROBLEM!!!')
                              new_filename_save = 'rnd2_' + METHOD + '_inv_problem_' + basename + '.yaml'
                              explanations_multi.create_problem_yaml(new_problem, new_filename_save)
                              generated = True
                              break

