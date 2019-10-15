import sys, os
sys.path.insert(0, 'X:\\Nicola_Gritti\\Repos\\tgmm_quality\\src')
import utils_registration as reg
import utils as ut
import numpy as np
from scipy.optimize import minimize, root_scalar

keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']

### load new data
cells = np.loadtxt(os.path.join('GMEMtracking3D_2019_10_14_16_22_2','cell_track_corr.txt'))
cells = cells[~np.isnan(cells).any(axis=1)]
embryos = np.loadtxt(os.path.join('GMEMtracking3D_2019_10_14_16_22_2','embryo_track_corr.txt'))

times = list(set(cells[:,keys.index('timepoint')]))

def rotation(params,pos_now,pos_prev):
    (phi,theta,psi) = params

    R = reg.rot_mat(phi,theta,psi)
    # print(R)

    dist = np.transpose(pos_prev) - np.matmul(R,np.transpose(pos_now))
    dist = np.linalg.norm(dist, axis=0)
    # print(dist.shape)
    # print(np.sum(dist))

    return np.sum(dist)

data = [[0,0,0,0]]
for t in times[1:]:
    print('#'*20,t,'\n')

    # load previous and current cells
    cells_now = cells[cells[:,keys.index('timepoint')]==t,:]
    cells_prev = cells[cells[:,keys.index('timepoint')]==(t-1),:]
    
    # filter for cells that appear in both previous and current timepoint (i.e. tracked correctly)
    cells_ids_now = cells_now[:,keys.index('cell_id')]
    cells_ids_prev = cells_prev[:,keys.index('cell_id')]
    cells_ids_to_keep = [ c for c in cells_ids_now if c in cells_ids_prev ]
    index = [ c in cells_ids_to_keep for c in cells_now[:,keys.index('cell_id')] ]
    cells_now = cells_now[index,:]
    index = [ c in cells_ids_to_keep for c in cells_prev[:,keys.index('cell_id')] ]
    cells_prev = cells_prev[index,:]

    # sort prev and current so that index match cells
    index = np.argsort(cells_now[:,keys.index('cell_id')])
    cells_now = cells_now[index]
    index = np.argsort(cells_prev[:,keys.index('cell_id')])
    cells_prev = cells_prev[index]
    # print(cells_now[:,keys.index('cell_id')])
    # print(cells_prev[:,keys.index('cell_id')])

    index = np.array([(k=='X')or(k=='Y')or(k=='Z') for k in keys])
    pos_now = cells_now[:,index]
    pos_prev = cells_prev[:,index]
    # print(pos_now.shape, pos_prev.shape)
    # print(pos_now[:5])
    # print(pos_prev[:5])

    phi = 0
    theta = 0
    psi = 0

    r = minimize(rotation,(phi,theta,psi),args=(pos_now,pos_prev))
    print(r.x)
    data.append([t,r.x[0],r.x[1],r.x[2]])
np.savetxt('piv_mat.txt',data, fmt='%i %1.10f  %1.10f  %1.10f')
    


