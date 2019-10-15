import xml.etree.ElementTree as ET
import os, glob, sys
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import rc
sys.path.insert(0, 'X:\\Nicola_Gritti\\Repos\\tgmm_quality\\src')
import utils as ut
plt.rcParams.update({'font.size': 20})
# plt.style.use('dark_background')
rc('pdf', fonttype=42)

path = os.path.join('XML_finalResult_lht')
keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']

### LOAD DATA
data_raw = np.loadtxt('cell_track_not_corr.txt')
lasttp = np.max(data_raw[:,keys.index('timepoint')])
data_raw = data_raw[data_raw[:,keys.index('timepoint')]!=lasttp,:]
data_raw[:,keys.index('timepoint')] += 20

### TRACK LENGTH
fig, ax = plt.subplots(1,1,figsize=(8,4))
fig.subplots_adjust(right=0.97,top=0.95,left=0.13,bottom=0.18)
ax.set_xlabel('Track length (timepoints)')
ax.set_ylabel('Number of lineages')
# set_white_plot(ax)
track_length(data_raw, ax=ax, color='tab:blue')

plt.show()
