import xml.etree.ElementTree as ET
import os, glob
import numpy as np
import json
from tifffile import imread,imsave
import matplotlib.pyplot as plt

def extract_divisions(data):
    timepoints = list(set(data[:,keys.index('timepoint')]))
    timepoints = timepoints[:-1]

    divisions = []
    for tp in timepoints:
        tp_data = data[data[:,keys.index('timepoint')]==tp]
        next_data = data[data[:,keys.index('timepoint')]==(tp+1)]

        new_cells = next_data[:,keys.index('cell_id')]
        new_parent_id = next_data[:,keys.index('parent_id')]
        for cell in tp_data:
            cell_id = cell[keys.index('cell_id')]
            if cell_id not in new_cells:
                if cell_id in new_parent_id:
                    divisions.append([tp,int(cell[keys.index('X')]/2),
                                            int(cell[keys.index('Y')]/2),
                                            int(cell[keys.index('Z')]/10)])
    divisions = np.array(divisions)
    return divisions

def extract_lineage_data(data, lineage_id):
    lineage_data = data[data[:,keys.index('lineage')]==lineage_id,:]
    return lineage_data

keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']# data = xml2numpy(path)

### LOAD DATA
print('loading data...')
data = np.loadtxt(os.path.join('..','restored_results','TGMM_plots','file.txt'))[:-1]
lasttp = np.max(data[:,keys.index('timepoint')])
data = data[data[:,keys.index('timepoint')]!=lasttp,:]

print('extracting divisions')
divisions = extract_divisions(data)
print(divisions.shape)

timepoints = list(set(divisions[:,0]))
print(timepoints)
for tp in timepoints:
    print(tp)
    movie = np.zeros((135,526,659)).astype(np.uint8)
    tp_div = divisions[divisions[:,0]==tp,1:]
    for div in tp_div:
        print(div)
        movie[int(div[2]-1):int(div[2]+2),
                int(div[1]-2):int(div[1]+3),
                int(div[0]-2):int(div[0]+3)] = 128

    movie.shape = (1,movie.shape[0],1,movie.shape[1],movie.shape[2],1)
    imsave('div_flashes\movie%05d.tif'%tp,movie, imagej=True)

