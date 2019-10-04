import xml.etree.ElementTree as ET
import os, glob
import numpy as np
import json
from tifffile import imread,imsave
import matplotlib.pyplot as plt

def extract_lineage_data(data, lineage_id):
    lineage_data = data[data[:,keys.index('lineage')]==lineage_id,:]
    return lineage_data

keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']# data = xml2numpy(path)

### LOAD DATA
data = np.loadtxt(os.path.join('..','restored_results','TGMM_plots','file.txt'))[:-1]
lasttp = np.max(data[:,keys.index('timepoint')])
data = data[data[:,keys.index('timepoint')]!=lasttp,:]

### SELECT LINEAGE TO CROP: DEEP IN THE SAMPLE
tp_data = data[data[:,keys.index('timepoint')]==15,:]
cm = np.array([np.mean(tp_data[:,keys.index('X')]),np.mean(tp_data[:,keys.index('Y')]),np.mean(tp_data[:,keys.index('Z')])])
dists = []
for c in tp_data:
    pos = np.array([c[keys.index('X')],c[keys.index('Y')],c[keys.index('Z')]])
    dists.append(np.sqrt(np.sum(((cm-pos)*0.6)**2)))
lineages = list(set(tp_data[(np.array(dists)>250)&(np.array(dists)<300),keys.index('lineage')]))
print(lineages)
print(len(tp_data),len(lineages))


lineage = 0
lineage_length = 0

for l in lineages:
    lineage_data = extract_lineage_data(data, l)
    lineage_length_new = len(list(set(lineage_data[:,keys.index('timepoint')])))
    if lineage_length_new>lineage_length:
        lineage = l
        lineage_length = lineage_length_new

lineage_data = extract_lineage_data(data, lineage)
cell_num = []
for t in list(set(lineage_data[:,keys.index('timepoint')])):
    cell_num.append(len(lineage_data[lineage_data[:,keys.index('timepoint')]==t,keys.index('cell_id')]))
    print(lineage_data[lineage_data[:,keys.index('timepoint')]==t,keys.index('cell_id')])
max_cells = np.max(cell_num)


hs = 25

# cell_ids = list(set(lineage_data[:,keys.index('cell_id')]))
# cell_ids.sort()
movie1 = np.zeros((lineage_length,50,50*max_cells))
movie2 = np.zeros((lineage_length,50,50*max_cells))

t_idx = 0
for t in list(set(lineage_data[:,keys.index('timepoint')])):
    print('timepoint',t)
    img1 = imread('..\\restoredScaled\\T%05d.tif'%t)
    img2 = imread('..\\raw\\pescoid1--C00--T%05d.tif'%t)
    tp_data = lineage_data[lineage_data[:,keys.index('timepoint')]==t,:]

    tp_img1 = []
    tp_img2 = []
    for cell in tp_data:
        c_id = cell[keys.index('cell_id')]
        pos = [int(cell[6]),int(cell[7]),int(cell[8]/5)]

        # cell1 = np.mean(img1[pos[2]-1:pos[2]+2,pos[1]-hs:pos[1]+hs,pos[0]-hs:pos[0]+hs],0)
        cell1 = img1[pos[2],pos[1]-hs:pos[1]+hs,pos[0]-hs:pos[0]+hs]
        tp_img1.append(cell1)

        # cell2 = np.mean(img2[pos[2]-1:pos[2]+2,pos[1]-hs:pos[1]+hs,pos[0]-hs:pos[0]+hs],0)
        cell2 = img2[pos[2],pos[1]-hs:pos[1]+hs,pos[0]-hs:pos[0]+hs]
        tp_img2.append(cell2)
        
    tp_img1 = np.concatenate([t for t in tp_img1]).transpose()
    tp_img2 = np.concatenate([t for t in tp_img2]).transpose()
    movie1[t_idx,:,:tp_img1.shape[1]] = tp_img1
    movie2[t_idx,:,:tp_img2.shape[1]] = tp_img2
    t_idx += 1

    imsave('movie1.tif'%cell,np.array(movie1).astype(np.uint16), photometric='minisblack')
    imsave('movie2.tif'%cell,np.array(movie2).astype(np.uint16), photometric='minisblack')
