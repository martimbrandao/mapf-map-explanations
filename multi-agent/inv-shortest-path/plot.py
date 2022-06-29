import os
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate
from path import *
import seaborn as sns
plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
markersize = 8
markers = ['o','v','s','p','*','x','D','d','+','.','h']
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

### params

files = ['bench-results-20220404-123117-full.csv', 'bench-results-20220401-163028-partial.csv', 'bench-results-20220404-125852-partial.csv']
organize_by_family = True # organize scatter plot by family? or same marker for all families of problems?
nticks = 5
failurelinelabel = 'unsolv.'
print_table_mode = 'latex' # 'text' or 'latex'

### print table

def printTable(table, table_headers):
  if print_table_mode == 'latex':
    headersL = [h+' &' for h in table_headers]
    tableL = [[str(a)+' &' for a in line] for line in table]
    headersL[-1] = headersL[-1][:-1] + ' \\\\'
    for line in tableL:
      line[-1] = line[-1][:-1] + ' \\\\'
    table_str = tabulate(tableL, headers=headersL, tablefmt="plain")
  elif print_table_mode == 'text':
    table_str = tabulate(table, headers=table_headers)
  title_str = 'Summary results ' + f
  print('')
  print('-' * table_str.find('\n'))
  print(' ' * int(table_str.find('\n')/2 - len(title_str)/2) + title_str)
  print('')
  print(table_str)
  print('')

### main

for f in files:
  data = np.genfromtxt(f, delimiter=',', dtype='str', comments='!')
  headers = data[0,:]
  stats = data[1:,:]

  # skip certain families of problems
  for fam in ['rnd_global', 'rnd_multi', 'rnd_single']:
    prob_families = np.array([prob[:prob.find('inv')-1] for prob in stats[:,0]])
    stats = stats[prob_families != fam, :]
  prob_families = np.array([prob[:prob.find('inv')-1] for prob in stats[:,0]])

  # show table
  table_str = tabulate(stats, headers=headers)
  title_str = 'Results '  + f
  print('')
  print('-' * table_str.find('\n'))
  print(' ' * int(table_str.find('\n')/2 - len(title_str)/2) + title_str)
  print('')
  print(table_str)
  print('')

  ### aggregate stats
  table = []
  table2 = []

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
    table.append( [family, str(search_suc)+'/'+str(len(idxs)), str(jsingle_suc)+'/'+str(len(idxs)), str(incr_suc)+'/'+str(len(idxs)), round(search_suc_time,2), round(jsingle_suc_time,2), round(incr_suc_time,2)] )
    table2.append( [family, str(search_suc)+'/'+str(len(idxs)), str(jsingle_suc)+'/'+str(len(idxs)), str(incr_suc)+'/'+str(len(idxs)), str(ijs_suc)+'/'+str(len(idxs)), round(search_suc_time,2), round(jsingle_suc_time,2), round(incr_suc_time,2), round(ijs_suc_time,2)] )

  # get stats over all families
  if True:
    family = 'all'
    idxs = np.arange(stats.shape[0])

    # incr
    incr_i = np.where(stats[:,4]=='True')
    incr_suc = len(incr_i[0])
    incr_suc_len = np.mean( stats[incr_i,5].astype('float') )
    incr_suc_time = np.mean( stats[incr_i,6].astype('float') )

    # jsingle
    jsingle_i = np.where(stats[:,13]=='True')
    jsingle_suc = len(jsingle_i[0])
    jsingle_suc_len = np.mean( stats[jsingle_i,14].astype('float') )
    jsingle_suc_time = np.mean( stats[jsingle_i,15].astype('float') )

    # search
    search_i = np.where(stats[:,16]=='True')
    search_suc = len(search_i[0])
    search_suc_len = np.mean( stats[search_i,17].astype('float') )
    search_suc_time = np.mean( stats[search_i,18].astype('float') )

    # incr-jsingle-search (faster and higher success rater, but sub-optimal)
    ijs_suc = 0
    ijs_suc_len = []
    ijs_suc_time = []
    for i in range(stats.shape[0]):
      if stats[i,4] == 'True':
        ijs_suc += 1
        ijs_suc_len.append( float(stats[i,5]) )
        ijs_suc_time.append( float(stats[i,6]) )
      elif stats[i,13] == 'True':
        ijs_suc += 1
        ijs_suc_len.append( float(stats[i,14]) )
        ijs_suc_time.append( float(stats[i,6]) + float(stats[i,15]) )
      elif stats[i,16] == 'True':
        ijs_suc += 1
        ijs_suc_len.append( float(stats[i,17]) )
        ijs_suc_time.append( float(stats[i,6]) + float(stats[i,15]) + float(stats[i,18]) )
    ijs_suc_len = np.mean(np.array(ijs_suc_len).astype('float'))
    ijs_suc_time = np.mean(np.array(ijs_suc_time).astype('float'))

    # table
    table.append( [family, str(search_suc)+'/'+str(len(idxs)), str(jsingle_suc)+'/'+str(len(idxs)), str(incr_suc)+'/'+str(len(idxs)), round(search_suc_time,2), round(jsingle_suc_time,2), round(incr_suc_time,2)] )
    table2.append( [family, str(search_suc)+'/'+str(len(idxs)), str(jsingle_suc)+'/'+str(len(idxs)), str(incr_suc)+'/'+str(len(idxs)), str(ijs_suc)+'/'+str(len(idxs)), round(search_suc_time,2), round(jsingle_suc_time,2), round(incr_suc_time,2), round(ijs_suc_time,2)] )

  # print table
  printTable(table, ['Problems','Search','JSingle','Incr','Search','JSingle','Incr'])
  printTable(table2, ['Problems','Search','JSingle','Incr','Collection','Search','JSingle','Incr','Collection'])

  # scatter plot incr vs search
  for typ in ['time', 'len']:

    for sig in ['incr', 'isp']:

      incr_i = np.where(stats[:,4]=='True')
      jsingle_i = np.where(stats[:,13]=='True')
      search_i = np.where(fstats[:,16]=='True')

      if typ == 'len':
        if sig == 'incr':
          x = stats[:,17].astype('float')
          y = stats[:,5].astype('float')
          max_xy = max(np.max(x) , np.max(y))
          failval1 = max_xy * 1.1
          x[stats[:,16]=='False'] = failval1
          y[stats[:,4]=='False'] = failval1
        elif sig == 'isp':
          x = stats[:,17].astype('float')
          y = stats[:,14].astype('float')
          max_xy = max(np.max(x) , np.max(y))
          failval1 = max_xy * 1.1
          x[stats[:,16]=='False'] = failval1
          y[stats[:,13]=='False'] = failval1
      elif typ == 'time':
        if sig == 'incr':
          x = stats[:,18].astype('float')
          y = stats[:,6].astype('float')
          max_xy = max(np.max(x) , np.max(y))
          failval1 = max_xy * 1.1
          x[stats[:,16]=='False'] = failval1
          y[stats[:,4]=='False'] = failval1
        elif sig == 'isp':
          x = stats[:,18].astype('float')
          y = stats[:,15].astype('float')
          max_xy = max(np.max(x) , np.max(y))
          failval1 = max_xy * 1.1
          x[stats[:,16]=='False'] = failval1
          y[stats[:,13]=='False'] = failval1

      failvalx = np.max(x[x < failval1]) * 1.1
      failvaly = np.max(y[y < failval1]) * 1.1

      # figure 1
      if False:
        fig, ax = plt.subplots()
        plt.plot([0,failval1], [failval1,failval1], color='#aaaaaa', linestyle='-')
        plt.plot([failval1,failval1], [0,failval1], color='#aaaaaa', linestyle='-')
        itmarker = itertools.cycle(markers)
        if organize_by_family:
          for family in np.unique(prob_families):
            idxs = np.where(prob_families == family)
            if len(idxs) == 1:
                idxs = idxs[0]
            marker = itmarker.next()
            plt.plot(x[idxs], y[idxs], color='black', marker=marker, markersize=markersize, linestyle='None', label=family)
        else:
          marker = itmarker.next()
          plt.plot(x, y, color='black', marker=marker, markersize=markersize, linestyle='None')

        # diagonal
        plt.plot([0,max_xy], [0,max_xy], color='black', linestyle='--')

        # text
        plt.legend(loc='upper left')
        plt.xlabel('Search Method')
        if sig == 'incr':
          plt.ylabel('Incremental Method')
        elif sig == 'isp':
          plt.ylabel('Joint ISP Method')

        # show/save
        plt.savefig(os.path.splitext(f)[0]+'-'+sig+'-'+typ+'-1.png', bbox_inches="tight")
        plt.show()

      # figure 2
      if False:
        fig, ax = plt.subplots()
        itmarker = itertools.cycle(markers)
        bothok = np.logical_and(x < failval1, y < failval1)
        max_xy = max(np.max(x[bothok]) , np.max(y[bothok]))
        marker = itmarker.next()
        plt.plot(x[bothok], y[bothok], color='black', marker=marker, markersize=markersize, linestyle='None')

        # text
        plt.legend(loc='upper left')
        plt.xlabel('Search Method')
        if sig == 'incr':
          plt.ylabel('Incremental Method')
        elif sig == 'isp':
          plt.ylabel('Joint ISP Method')

        # show/save
        plt.savefig(os.path.splitext(f)[0]+'-'+sig+'-'+typ+'-2.png', bbox_inches="tight")
        plt.show()

      # figure 3
      if True:
        fig, ax = plt.subplots()
        plt.plot([0,failvalx], [failvaly,failvaly], color='#aaaaaa', linestyle='-')
        plt.plot([failvalx,failvalx], [0,failvaly], color='#aaaaaa', linestyle='-')
        x[x==failval1] = failvalx
        y[y==failval1] = failvaly
        itmarker = itertools.cycle(markers)
        itcolor = itertools.cycle(colors)
        if organize_by_family:
          for family in np.unique(prob_families):
            idxs = np.where(prob_families == family)
            if len(idxs) == 1:
                idxs = idxs[0]
            marker = itmarker.next()
            color = itcolor.next().lstrip('#')
            color = tuple([float(int(color[i:i+2], 16))/255 for i in (0, 2, 4)] + [0.5])
            plt.plot(x[idxs], y[idxs], marker=marker, markersize=markersize, color=color, markerfacecolor=color, linestyle='None', label=family)
        else:
          marker = itmarker.next()
          plt.plot(x, y, marker=marker, markersize=markersize, color=(0, 0, 0, 0.5), markerfacecolor=(0, 0, 0, 0.5), linestyle='None')

        # diagonal
        plt.plot([0,min(failvalx,failvaly)], [0,min(failvalx,failvaly)], color='black', linestyle='--')

        # text
        plt.legend(loc=(0.1, 0.7))
        if typ == 'time':
          plt.xlabel('Search method time (s)')
        elif typ == 'len':
          plt.xlabel('Search method expl. length')
        if sig == 'incr':
          if typ == 'time':
            plt.ylabel('Incremental method time (s)')
          elif typ == 'len':
            plt.ylabel('Incremental method expl. length')
        elif sig == 'isp':
          if typ == 'time':
            plt.ylabel('Joint ISP method time (s)')
          elif typ == 'len':
            plt.ylabel('Joint ISP method expl. length')

        # ticks
        xticks = np.linspace(0, round(failvalx), nticks)
        yticks = np.linspace(0, round(failvaly), nticks)
        xticklabels = np.round(xticks,1).astype(str)
        yticklabels = np.round(yticks,1).astype(str)
        xticks[-1] = failvalx
        yticks[-1] = failvaly
        xticklabels[-1] = failurelinelabel
        yticklabels[-1] = failurelinelabel
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        # show/save
        plt.tight_layout(rect=[0,0,1,1])
        plt.savefig(os.path.splitext(f)[0]+'-'+sig+'-'+typ+'-3.png', bbox_inches="tight")
        plt.show()

