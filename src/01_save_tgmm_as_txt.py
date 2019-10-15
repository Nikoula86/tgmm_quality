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
keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']# data = xml2numpy(path)

data = ut.xml2numpy(path)

### LOAD DATA
# data_raw = np.loadtxt('cell_track_not_corr.txt')
# lasttp = np.max(data_raw[:,keys.index('timepoint')])
# data_raw = data_raw[data_raw[:,keys.index('timepoint')]!=lasttp,:]

# data_rec = np.loadtxt('..\\restored_results\\TGMM_plots\\file.txt')
# lasttp = np.max(data_rec[:,keys.index('timepoint')])
# data_rec = data_rec[data_rec[:,keys.index('timepoint')]!=lasttp,:]

### CELL NUMBER OVER TIME
# fig, ax = plt.subplots(1,1,figsize=(8,4))
# fig.subplots_adjust(right=0.97,top=0.95,left=0.13,bottom=0.18)
# ax.set_xlabel('Timepoint')
# ax.set_ylabel('Number of cells')
# # set_white_plot(ax)
# n_cells_raw, _ = cell_number(data_raw, ax=ax, color='tab:blue')
# # n_cells_rec, _ = cell_number(data_rec, ax=ax, color='tab:orange')
# # print(np.mean((n_cells_rec-n_cells_raw)/n_cells_raw))
# # print(np.std((n_cells_rec-n_cells_raw)/n_cells_raw))

# ### RADIAL DISTRIBUTION OF CELLS
# fig, ax = plt.subplots(1,1,figsize=(8,4))
# fig.subplots_adjust(right=0.97,top=0.95,left=0.13,bottom=0.18)
# ax.set_xlabel('Radial distance from c.m. (um)')
# ax.set_ylabel('Number of cells')
# set_white_plot(ax)
# radial_distribution(data_rec, ax=ax, color='tab:orange')
# radial_distribution(data_raw, ax=ax, color='tab:blue')

# ### sanity check for cm calculation
# # tp_data = data_rec[data_rec[:,keys.index('timepoint')]==15,:]
# # cm = [np.mean(tp_data[:,keys.index('X')]),np.mean(tp_data[:,keys.index('Y')]),np.mean(tp_data[:,keys.index('Z')])]
# # print(cm)
# # radial_distribution(data_raw, cm=cm, ax=ax, color='lightblue')

# # tp_data = data_raw[data_raw[:,keys.index('timepoint')]==15,:]
# # cm = [np.mean(tp_data[:,keys.index('X')]),np.mean(tp_data[:,keys.index('Y')]),np.mean(tp_data[:,keys.index('Z')])]
# # print(cm)
# # radial_distribution(data_rec, cm=cm, ax=ax, color='red')

### TRACK LENGTH
# fig, ax = plt.subplots(1,1,figsize=(8,4))
# fig.subplots_adjust(right=0.97,top=0.95,left=0.13,bottom=0.18)
# ax.set_xlabel('Track length (timepoints)')
# ax.set_ylabel('Number of lineages')
# # set_white_plot(ax)
# # track_length(data_rec, ax=ax, color='tab:orange')
# track_length(data_raw, ax=ax, color='tab:blue')

# plt.show()
