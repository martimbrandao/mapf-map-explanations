import glob
import time
import tabulate
import explanations_multi
import explanations_multi_incremental
import explanations_global
import explanations_global_insisting_single
import baseline_single_agent
import baseline_joint_single_agent
import baseline_search
from path import *
import numpy as np
import pdb

ANIMATE = False
QUESTION_PARTIAL_PLAN = False

# TODO:
# - add new baseline for full-plan-question: "joint single" (multiple single-agent inv-mapf but all agents in one opt problem; no obstacles on top of desired solution)
# - (optional: only if we want to benchmark "multi") constrained-solve should receive argument to set whether waiting is allowed on full-path waypoint lists; I guess "multi" only has 100% success rate when waits are not allowed; and it has no advantage over "incr"... so maybe we can just avoid talking about multi ? some insights to be taken though...

# TODO:
# - checking if solution satisfies desired paths should ignore waits (in is_mapf_solution_valid)
# - double check that CBS is optimizing cost, not makespan
#
# NOTES:
# there are some differences between what the methods (can) do, hence comparison unfair:
#   - multi assumes desired paths have to occur exactly (without adding waits); but incr lets waits occur (because Areeb's constrained CBS works that way)
#   - multi makes each desired path optimal given that other paths are fixed; but does not guarantee [cost|desired,fixed,new_obstacles] == [cost|new_obstacles]
#   - incr guarantees [cost|desired,new_obstacles] == [cost|new_obstacles]; but not [cost|desired,fixed,new_obstacles] == [cost|new_obstacles]
#          double check above actually what does it guarantee??????????
# so to be fair we should have multiple versions of incr:
#   - incr1) comparable to multi, hence should: 
#            - guarantee [cost|desired,fixed,new_obstacles] == [cost|new_obstacles]  ----> so it is going to be better than multi (which does == [cost|fixed,new_obstacles])
#            - does not let waits occur                                                    i.e. solve some problems that multi can't 
#                                                                                          but multi might also solve some problems that incr can't (why??)
#            - need to change sanity check to double check that [cost|desired,fixed,new_obstacles] == [cost|new_obstacles]
#   - incr2) as 'global' as possible, hence should:
#            - guarantee [cost|desired,new_obstacles] == [cost|new_obstacles]
#            - let waits occur (?)                                                   ----> this is what we already have now; different kind of explanations
#                                                                                          not comparable to multi, comparable to single without [OtherPaths=Free] constraint
#            - sanity check as it is now, so making sure [cost|desired,new_obstacles] == [cost|new_obstacles]
#
# RETHINK EVERYTHING
#
# - problem 1: "plan-optimality" question - "why not complete solution X?", where X = [paths D for agents A + previous paths for other agents]
#                - obstacles valid if cost(CBS(obst)) == cost(X)
#                - waits are not allowed on top of D
#              algos:
#              - multi  (done)  [as is; not sure it is complete]
#              - incr   (done)  [obst such that cost(badX_i) > cost(X); stop once cost(CBS(obst)) == cost(X)]
#              - single (done)  [run single on each agent, take union, hope it works]
#              - search (done)  [skip combinations with obst in X, for layer N (with N obstacles) only consider adding them in locations of A-paths seen on layers up to N-1]
#
# - problem 2: "partial-plan-optimality" question "why not agents A doing paths D?"
#                - obstacles valid if cost(CBS(obst)) == cost(CBS(obst,D))
#                - we allow extra waits on D here (b/c based on user study we care about paths only)
#              algos:
#              - incr   (done)  [obst such that cost(badX_i) > cost(CBS(obst,D)); stop on cost condition; not complete?]
#              - global (done)  [obst such that cost(badX_i) > cost(Y); where Y in [CBS(obst_i,D), where obst_i are sols we find along the way]; stop on cost condition; not complete?]
#              - single (done)  [run single on each agent, take union, hope it works]
#              - search (done)  [skip combinations with obst in D, for layer N (with N obstacles) only consider adding them in locations of paths seen on layers up to N-1]
#


# Main
if __name__ == '__main__':

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if QUESTION_PARTIAL_PLAN:
        outfilename = 'bench-results-'+timestr+'-partial.csv'
    else:
        outfilename = 'bench-results-'+timestr+'-full.csv'

    files = []
    if QUESTION_PARTIAL_PLAN:
        files += sorted(glob.glob(ROOT_PATH + '/rnd_problems_2022/partial_plan_questions/incr/*.yaml'))
        files += sorted(glob.glob(ROOT_PATH + '/rnd_problems_2022/partial_plan_questions/jsingle/*.yaml'))
        files += sorted(glob.glob(ROOT_PATH + '/rnd_problems_2022/partial_plan_questions/search/*.yaml'))
    else:
        files += sorted(glob.glob(ROOT_PATH + '/rnd_problems_2022/full_plan_questions/incr/*.yaml'))
        files += sorted(glob.glob(ROOT_PATH + '/rnd_problems_2022/full_plan_questions/jsingle/*.yaml'))
        files += sorted(glob.glob(ROOT_PATH + '/rnd_problems_2022/full_plan_questions/search/*.yaml'))

    problems = []

    for f in files:

        print(f)

        # original solution
        raw_problem = explanations_multi.parse_yaml(f)
        raw_obst = raw_problem['map']['obstacles']

        # solve with multi-agent version
        print('Starting Multi...')
        t1 = time.time()
        try:
            #if QUESTION_PARTIAL_PLAN == False:
            #    success_multi, obst_multi = explanations_multi.main_inv_mapf_fullpath(f, False, ANIMATE)
            #else:
            #    success_multi = False
            #    obst_multi = []
            success_multi = False
            obst_multi = []
        except KeyError:
            success_multi = False
            obst_multi = []
        t2 = time.time()
        time_multi  = round(t2 - t1, 2)
        len_multi = max(0, len(obst_multi) - len(raw_obst))
        print('Multi: ' + str(success_multi))

        # solve with multi-agent INCREMENTAL version
        print('Starting Incr...')
        t1 = time.time()
        try:
            success_incr, obst_incr = explanations_multi_incremental.main_inv_mapf_fullpath(f, False, ANIMATE, QUESTION_PARTIAL_PLAN)
        except KeyError:
            success_incr = False
            obst_incr = []
        t2 = time.time()
        time_incr  = round(t2 - t1, 2)
        len_incr = max(0, len(obst_incr) - len(raw_obst))
        print('Incr: ' + str(success_incr))

        # solve with multi-agent GLOBAL version
        print('Starting Global...')
        t1 = time.time()
        try:
            #success_global, obst_global = explanations_global.main_inv_mapf_fullpath(f, False, ANIMATE, QUESTION_PARTIAL_PLAN)
            success_global = False
            obst_global = []
        except KeyError:
            success_global = False
            obst_global = []
        t2 = time.time()
        time_global  = round(t2 - t1, 2)
        len_global = max(0, len(obst_global) - len(raw_obst))
        print('Global: ' + str(success_global))

        # solve with single-agent version
        print('Starting Single...')
        t1 = time.time()
        try:
            #success_single, obst_single = baseline_single_agent.main_inv_mapf_fullpath(f, False, ANIMATE, QUESTION_PARTIAL_PLAN)
            success_single = False
            obst_single = []
        except KeyError:
            success_single = False
            obst_single = []
        t2 = time.time()
        time_single  = round(t2 - t1, 2)
        len_single = max(0, len(obst_single) - len(raw_obst))
        print('Single: ' + str(success_single))

        # solve with joint single-agent version
        print('Starting JointSingle...')
        t1 = time.time()
        try:
            success_joint_single, obst_joint_single = baseline_joint_single_agent.main_inv_mapf_fullpath(f, False, ANIMATE, QUESTION_PARTIAL_PLAN)
        except KeyError:
            success_joint_single = False
            obst_joint_single = []
        t2 = time.time()
        time_joint_single  = round(t2 - t1, 2)
        len_joint_single = max(0, len(obst_joint_single) - len(raw_obst))
        print('JointSingle: ' + str(success_joint_single))

        # solve with search-based version
        print('Starting Search...')
        t1 = time.time()
        try:
            success_search, obst_search = baseline_search.main_inv_mapf_fullpath(f, False, ANIMATE, QUESTION_PARTIAL_PLAN)
        except KeyError:
            success_search = False
            obst_search = []
        t2 = time.time()
        time_search  = round(t2 - t1, 2)
        len_search = max(0, len(obst_search) - len(raw_obst))
        print('Search: ' + str(success_search))

        # save stats
        problems.append( [os.path.basename(f), success_multi, len_multi, time_multi, success_incr, len_incr, time_incr, success_global, len_global, time_global, success_single, len_single, time_single, success_joint_single, len_joint_single, time_joint_single, success_search, len_search, time_search] )

        headers = ['Problem','So-M','#-M','t-M','So-I','#-I','t-I','So-G','#-G','t-G','So-S','#-S','t-S','So-JS','#-JS','t-JS','So-*','#-*','t-*']

        # save stats
        results = np.array([headers] + problems)
        np.savetxt(outfilename, results, delimiter=",", fmt='%s')

        # show table so far
        table_str = tabulate.tabulate(problems, headers=headers)
        title_str = 'Results so far'
        print('')
        print('-' * table_str.find('\n'))
        print(' ' * int(table_str.find('\n')/2 - len(title_str)/2) + 'Results so far')
        print('')
        print(table_str)
        print('')

        ### aggregate stats
        stats = np.array(problems)
        table = []

        # get families of problems
        prob_families = np.array([prob[:prob.find('inv')-1] for prob in stats[:,0]])

        # get stats for each family of problems
        for family in np.unique(prob_families):

            idxs = np.where(prob_families == family)
            if len(idxs) == 1:
                idxs = idxs[0]
            fstats = stats[idxs, :]

            # incr
            incr_i = np.where(fstats[:,4]=='True')
            incr_suc = len(incr_i[0])
            incr_suc_len = np.mean( fstats[incr_i,5].astype('float') )
            incr_suc_time = np.mean( fstats[incr_i,6].astype('float') )

            # jsingle
            jsingle_i = np.where(fstats[:,13]=='True')
            jsingle_suc = len(jsingle_i[0])
            jsingle_suc_len = np.mean( fstats[jsingle_i,14].astype('float') )
            jsingle_suc_time = np.mean( fstats[jsingle_i,15].astype('float') )

            # search
            search_i = np.where(fstats[:,16]=='True')
            search_suc = len(search_i[0])
            search_suc_len = np.mean( fstats[search_i,17].astype('float') )
            search_suc_time = np.mean( fstats[search_i,18].astype('float') )

            # incr-jsingle-search (faster and higher success rater, but sub-optimal)
            ijs_suc = 0
            ijs_suc_len = []
            ijs_suc_time = []
            for i in range(fstats.shape[0]):
                if fstats[i,4] == 'True':
                    ijs_suc += 1
                    ijs_suc_len.append( float(fstats[i,5]) )
                    ijs_suc_time.append( float(fstats[i,6]) )
                elif fstats[i,13] == 'True':
                    ijs_suc += 1
                    ijs_suc_len.append( float(fstats[i,14]) )
                    ijs_suc_time.append( float(fstats[i,6]) + float(fstats[i,15]) )
                elif fstats[i,16] == 'True':
                    ijs_suc += 1
                    ijs_suc_len.append( float(fstats[i,17]) )
                    ijs_suc_time.append( float(fstats[i,6]) + float(fstats[i,15]) + float(fstats[i,18]) )
            ijs_suc_len = np.mean(np.array(ijs_suc_len).astype('float'))
            ijs_suc_time = np.mean(np.array(ijs_suc_time).astype('float'))

            # table
            table.append( [family, incr_suc, incr_suc_len, incr_suc_time, jsingle_suc, jsingle_suc_len, jsingle_suc_time, search_suc, search_suc_len, search_suc_time, ijs_suc, ijs_suc_len, ijs_suc_time] )

        # print table        
        table_str = tabulate.tabulate(table, headers=['Problems','So-I','#-I','t-I','So-JS','#-JS','t-JS','So-*','#-*','t-*','So-All','#-All','t-All'])
        title_str = 'Results so far'
        print('')
        print('-' * table_str.find('\n'))
        print(' ' * int(table_str.find('\n')/2 - len(title_str)/2) + title_str)
        print('')
        print(table_str)
        print('')

