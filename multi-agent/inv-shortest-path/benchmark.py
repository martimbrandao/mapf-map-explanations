import glob
import time
import tabulate
import explanations_multi
import explanations_multi_incremental
import baseline_single_agent
from path import *
import pdb

ANIMATE = False

# Main
if __name__ == '__main__':

    problems = []

    for f in sorted(glob.glob(ROOT_PATH + '/rnd_problems_incr/*.yaml')):

        print(f)

        #if f != '/home/k1929632/workspace/martim/xaip/invmapf/multi-agent/inv-shortest-path/rnd_problems_single/rnd_single_inv_problem_map_8by8_obst12_agents5_ex21.yaml':
        #    continue

        # original solution
        raw_problem = explanations_multi.parse_yaml(f)
        raw_obst = raw_problem['map']['obstacles']

        # solve with multi-agent version
        t1 = time.time()
        try:
            success_multi, obst_multi = explanations_multi.main_inv_mapf_fullpath(f, False, ANIMATE)
        except KeyError:
            success_multi = False
            obst_multi = []
        t2 = time.time()
        time_multi  = t2 - t1
        len_multi = max(0, len(obst_multi) - len(raw_obst))
        print('Multi: ' + str(success_multi))

        # solve with multi-agent INCREMENTAL version
        t1 = time.time()
        try:
            success_incr, obst_incr = explanations_multi_incremental.main_inv_mapf_fullpath(f, False, ANIMATE)
        except KeyError:
            success_incr = False
            obst_incr = []
        t2 = time.time()
        time_incr  = t2 - t1
        len_incr = max(0, len(obst_incr) - len(raw_obst))
        print('Incr: ' + str(success_incr))

        # solve with single-agent version
        t1 = time.time()
        try:
            success_single, obst_single = baseline_single_agent.main_inv_mapf_fullpath(f, False, ANIMATE)
        except KeyError:
            success_single = False
            obst_single = []
        t2 = time.time()
        time_single  = t2 - t1
        len_single = max(0, len(obst_single) - len(raw_obst))
        print('Single: ' + str(success_single))

        # save stats
        problems.append( [os.path.basename(f), success_multi, len_multi, time_multi, success_incr, len_incr, time_incr, success_single, len_single, time_single] )

        # show table so far
        table_str = tabulate.tabulate(problems, headers=['Problem','Suc-M','Obs-M','Time-M','Suc-I','Obs-I','Time-I','Suc-S','Obs-S','Time-S'])
        title_str = 'Results so far'
        print('')
        print('-' * table_str.find('\n'))
        print(' ' * int(table_str.find('\n')/2 - len(title_str)/2) + 'Results so far')
        print('')
        print(table_str)
        print('')

