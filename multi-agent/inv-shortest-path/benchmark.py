import glob
import tabulate
import explanations_multi
import baseline_single_agent
from path import *
import pdb

ANIMATE = False

# Main
if __name__ == '__main__':

    problems = []

    for f in sorted(glob.glob(ROOT_PATH + '/rnd_problems_single/*.yaml')):

        print(f)

        #if f != '/home/k1929632/workspace/martim/xaip/invmapf/multi-agent/inv-shortest-path/rnd_problems_single/rnd_single_inv_problem_map_8by8_obst12_agents5_ex21.yaml':
        #    continue

        # original solution
        raw_problem = explanations_multi.parse_yaml(f)
        raw_obst = raw_problem['map']['obstacles']

        # solve with multi-agent version
        try:
            success_multi, obst_multi = explanations_multi.main_inv_mapf_fullpath(f, False, ANIMATE)
        except KeyError:
            success_multi = False
            obst_multi = []
        print('Multi: ' + str(success_multi))

        # solve with single-agent version
        try:
            success_single, obst_single = baseline_single_agent.main_inv_mapf_fullpath(f, False, ANIMATE)
        except KeyError:
            success_single = False
            obst_single = []
        print('Single: ' + str(success_single))

        problems.append( [os.path.basename(f), success_multi, success_single, max(0,len(obst_multi)-len(raw_obst)), max(0,len(obst_single)-len(raw_obst))] )

    print(tabulate.tabulate(problems, headers=['Problem','Success-Multi','Success-Single','#Obst-Multi','#Obst-Single']))
