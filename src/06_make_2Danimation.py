import numpy as np
import os, sys
from scipy.optimize import minimize, root_scalar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.filters import gaussian_filter1d
sys.path.insert(0, 'X:\\Nicola_Gritti\\Repos\\tgmm_quality\\src')
import utils_registration as reg

keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']

### load new data
cells = np.loadtxt(os.path.join('GMEMtracking3D_2019_10_14_16_22_2','cell_track_corr.txt'))
embryos = np.loadtxt(os.path.join('GMEMtracking3D_2019_10_14_16_22_2','embryo_track_corr.txt'))
# cells = cells[(cells[:,keys.index('timepoint')]<100),:]
print(cells.shape)
print(np.array([(k=='X')or(k=='Y')or(k=='Z') for k in keys]))
pos = cells[:,np.array([(k=='X')or(k=='Y')or(k=='Z') for k in keys])]
sph = reg.findSpherical(pos)
cells = np.append(cells,np.zeros((cells.shape[0],3)),axis=1)
keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore','r','elev','azim']
cells[:,keys.index('r')] = sph[:,0]
cells[:,keys.index('elev')] = sph[:,1]
cells[:,keys.index('azim')] = sph[:,2]

### filter for timepoints
_mintp = 70
_maxtp = 120
embryos = embryos[_mintp:_maxtp]
t_col = cells[:,keys.index('timepoint')]
t_idx = (t_col<_maxtp)&(t_col>=_mintp)
cells = cells[t_idx]
cells[:,keys.index('timepoint')] -= _mintp

def make_single_plot(t, ax, cells, embryos):
    ax.clear()
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)

    t_col = cells[:,keys.index('timepoint')]
    data_plot = cells[t_col==t,:]
    print(t, data_plot.shape)
    data_plot = data_plot[~np.isnan(data_plot).any(axis=1),:]

    ### find out color
    t_idx = (t_col<=t)
    data_tp = cells[t_idx,:]
    data_tp = data_tp[~np.isnan(data_tp).any(axis=1),:]
    # color = []
    # alpha = []
    d = []
    for cell_id in data_plot[:,keys.index('cell_id')]:
        data_cell = data_tp[data_tp[:,keys.index('cell_id')]==cell_id,:]

        # color.append( 'tab:blue' )
        # alpha.append(0.5)
        if data_cell.shape[0]>=3:
            start_pos = np.array([data_cell[-3,keys.index('X')],data_cell[-3,keys.index('Y')],data_cell[-3,keys.index('Z')]])
            final_pos = np.array([data_cell[-1,keys.index('X')],data_cell[-1,keys.index('Y')],data_cell[-1,keys.index('Z')]])
        #     start_embryos = embryos[t-2,1:4]
        #     final_embryos = embryos[t,1:4]
        #     start_radius = np.linalg.norm(start_pos - start_embryos)
        #     final_radius = np.linalg.norm(final_pos - final_embryos)
            d.append([data_cell[-1,keys.index('X')],data_cell[-1,keys.index('Y')],final_pos[2]-start_pos[2]])
    #         if (final_pos[2]>start_pos[2]):# & (final_radius<start_radius):
    #             color[-1] = 'tab:red'
    #             alpha[-1] = 1.
    # alpha = np.array(alpha)

    # print(t, data_plot.shape,len(color))
    # blu = [data_plot[alpha==.5,6],data_plot[alpha==.5,7],'tab:blue',.5]
    # ax.scatter(blu[0],blu[1],color=blu[2],alpha=.5, linewidth=0.)
    # red = [data_plot[alpha==1.,6],data_plot[alpha==1.,7],'tab:red',1.]
    # ax.scatter(red[0],red[1],color=red[2],alpha=1., linewidth=0.)

    d = np.array(d)
    if d.shape[0]>0:
        ax.scatter(d[:,0],d[:,1], c=d[:,2], vmin=-10, vmax=10)

    return


def make_animation(cells,embryos,name='epiboly_animation_2d.mp4',new_keys = ['timepoint','X','Y','Z','r','elev','dElev','dR']):
    ## plot in 3d vertical view
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    def init():
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        return

    _ani = animation.FuncAnimation(fig, make_single_plot, frames=int(len(embryos)), fargs = (ax, cells, embryos),
                                       interval=50, blit=False)
    Writer = animation.writers['ffmpeg']
    FFwriter = Writer(fps=5)
    _ani.save(name, writer = FFwriter, dpi=300)

print('starting animation...')
make_animation(cells,embryos,new_keys=keys,name='epiboly_animation_2d.mp4')

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111)
# ax.set_xlim(-500, 500)
# ax.set_ylim(-500, 500)
# make_single_plot(90, ax, cells, embryos)

plt.show()