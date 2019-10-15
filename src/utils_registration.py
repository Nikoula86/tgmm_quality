import numpy as np
import os, sys
from scipy.optimize import minimize, root_scalar
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

def extract_pos(data,timepoint,keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']):
    # filter single timepoint
    pos = data[data[:,keys.index('timepoint')]==timepoint,:]
    # filter 3D coordinates only
    index = np.array([(k=='X')or(k=='Y')or(k=='Z') for k in keys])
    pos = pos[:,index]
    pos = pos[~np.isnan(pos).any(axis=1),:]
    return pos

def fit_sphere(x0,pos3d):
    pos3d = np.append(pos3d,np.zeros((pos3d.shape[0],1)),axis=1)
    metric = np.eye(4)
    metric[-1,-1] = -1
    res = 0
    for p in pos3d:
        res += (np.matmul(np.matmul((x0-p),metric),(x0-p)))**2
    return res

def extract_cm(pos3d):
    cm = np.array(np.mean(pos3d,axis=0))
    return cm

def compute_centroid_and_cm(data_raw,_max=100,name='embryo_track.txt'):
    ### generate file
    data = []
    for t in np.arange(_max+1):
        pos = extract_pos(data_raw,t)
        r = minimize(fit_sphere,(0,0,0,500),args=pos)
        cm = extract_cm(pos)
        print(t, 'embryo (x,y,z,r):', r.x, 'cm_cell (x,y,z):', cm)
        data.append([t,r.x[0],r.x[1],r.x[2],r.x[3],cm[0],cm[1],cm[2]])
    data = np.array(data)

    np.savetxt(name,data, fmt='%i %1.4f  %1.4f  %1.4f  %1.4f  %1.4f  %1.4f  %1.4f')

def correct_pos_file(centroid,cm,data_raw,_max=100,name='cell_track.txt',keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']):
    data_raw = data_raw[data_raw[:,keys.index('timepoint')]<_max,:]

    exs,eys,ezs = generate_new_refsys(centroid,cm)
    
    for idx, raw in enumerate(data_raw):
        t = int(raw[keys.index('timepoint')])
        ez = ezs[t] # (cm[t]-centroid[t])/np.linalg.norm(cm[t]-centroid[t])
        # ex = np.array([ez[1], -ez[0], 0])
        ex = exs[t] # ex/np.linalg.norm(ex)
        # ey = np.cross(ez, ex)
        ey = eys[t] # ey/np.linalg.norm(ey)

        pos = raw[np.array([(k=='X')or(k=='Y')or(k=='Z') for k in keys])]
        pos = pos-centroid[t]
        pos = [np.dot(pos,ex),np.dot(pos,ey),np.dot(pos,ez)]
        data_raw[idx,keys.index('X')] = pos[0]
        data_raw[idx,keys.index('Y')] = pos[1]
        data_raw[idx,keys.index('Z')] = pos[2]
        
    np.savetxt(name,data_raw, fmt='%i %i %i %i %i %i %1.4f %1.4f %1.4f %i')

def plot_single3d(data,centroid,cm,t=99):
    ### plot in 3d - not corrected
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45,azim=45)

    ax.plot(centroid[:t,0], centroid[:t,1], centroid[:t,2],lw=2)
    ax.plot(cm[:t,0], cm[:t,1], cm[:t,2],lw=2)

    pos = extract_pos(data,t)
    ax.scatter(pos[:,0],pos[:,1],pos[:,2],alpha=0.2)

def make_3d_animation(data,centroid,cm,name='epiboly_animation.mp4',ang=[20,45],keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']):
    ## plot in 3d vertical view
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=ang[0],azim=ang[1])

    def init():
        ax.set_xlim(-500+np.mean(centroid[:,0]), 500+np.mean(centroid[:,0]))
        ax.set_ylim(-500+np.mean(centroid[:,1]), 500+np.mean(centroid[:,1]))
        ax.set_zlim(-500+np.mean(centroid[:,2]), 500+np.mean(centroid[:,2]))
        return

    def update(t, centroid, cm, data, keys):
        ax.clear()
        ax.set_xlim(-500+np.mean(centroid[:,0]), 500+np.mean(centroid[:,0]))
        ax.set_ylim(-500+np.mean(centroid[:,1]), 500+np.mean(centroid[:,1]))
        ax.set_zlim(-500+np.mean(centroid[:,2]), 500+np.mean(centroid[:,2]))

        t_col = data[:,keys.index('timepoint')]
        t_idx = (t_col<=t)&(t_col>(t-5))

        data_tp = data[t_idx,:]
        pos = extract_pos(data_tp,t)
        ax.scatter(pos[:,0],pos[:,1],pos[:,2],alpha=0.2, linewidth=0., color='tab:blue')

        ax.plot(centroid[:t,0], centroid[:t,1], centroid[:t,2],lw=2,color='tab:green')
        ax.plot(cm[:t,0], cm[:t,1], cm[:t,2],lw=2,color='tab:green')

        ax.quiver(centroid[t,0], centroid[t,1], centroid[t,2], 
                    cm[t,0]-centroid[t,0], cm[t,1]-centroid[t,1], cm[t,2]-centroid[t,2], 
                    length=1,
                    color='tab:orange' )
        
        times = [0,0]
        for cell_id in set(data_tp[:,keys.index('cell_id')]):
            start = time()
            data_cell = data_tp[data_tp[:,keys.index('cell_id')]==cell_id,:]
            l1 = time()
            times[0] += l1-start
            if data_cell.shape[0]==5:
                start_pos = [data_cell[-5,keys.index('Z')]]
                final_pos = [data_cell[-1,keys.index('Z')]]
                color = 'tab:blue'
                if final_pos > start_pos:
                    color = 'tab:red'
                ax.plot(data_cell[-5:,keys.index('X')],
                            data_cell[-5:,keys.index('Y')],
                            data_cell[-5:,keys.index('Z')],lw=1,color=color,alpha=.5)
            times[1] += time()-l1

        print(t, times)

        return

    _3dani = animation.FuncAnimation(fig, update, cm.shape[0], fargs=(centroid, cm, data, keys),
                                       interval=50, blit=False)
    Writer = animation.writers['ffmpeg']
    FFwriter = Writer(fps=10)
    _3dani.save(name, writer = FFwriter, dpi=300)

def findSpherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    # ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

###################################################################################################

def generate_meshgrid(ex,ey,ez,centroid,_range=(500,500,500),step=(1,1,1)):

    ox = np.array([1,0,0])
    oy = np.array([0,1,0])
    oz = np.array([0,0,1])

    M = np.array([[np.dot(ox,ex),np.dot(ox,ey),np.dot(ox,ez)],
                    [np.dot(oy,ex),np.dot(oy,ey),np.dot(oy,ez)],
                    [np.dot(oz,ex),np.dot(oz,ey),np.dot(oz,ez)]])

    start = time()
    g = np.meshgrid(np.arange(-int(_range[0]),int(_range[0]),step[0]),
                    np.arange(-int(_range[1]),int(_range[1]),step[1]),
                    np.arange(-int(_range[2]),int(_range[2]),step[2]))
    g = np.stack([g[0],g[1],g[2]])
    g_shape = g.shape[1:]
    g = np.reshape(g,(g.shape[0],np.prod(g.shape[1:])))
    print('\t\t generating meshgrid took:', time()-start)
    start = time()
    points = np.dot(M,g).T+centroid
    points = points[:,[2,1,0]].astype(np.float32)
    print('\t\t meshgrid shape:',points.shape)
    print('\t\t computing meshgrid in old ref syst took:', time()-start)

    M = None
    g = None

    return points,g_shape

def compute_rot_mat(d,th):
    ct = np.cos(th)
    st = np.sin(th)
    ct_d = 1-ct
    r1 = np.array( [ ct+d[0]*d[0]*ct_d, d[0]*d[1]*ct_d-d[2]*st, d[0]*d[2]*ct_d+d[1]*st ] )
    r2 = np.array( [ d[0]*d[1]*ct_d+d[2]*st, ct+d[1]*d[1]*ct_d, d[1]*d[2]*ct_d-d[0]*st ] )
    r3 = np.array( [ d[0]*d[2]*ct_d-d[1]*st, d[1]*d[2]*ct_d+d[0]*st, ct+d[2]*d[2]*ct_d ] )
    return np.stack([r1,r2,r3])

def generate_new_refsys(centroids,cms):

    ez = []
    for centroid, cm in zip(centroids, cms):
        ez.append( (cm-centroid)/np.linalg.norm(cm-centroid) )
    ez = np.array(ez)

    ### we use parallel transport of vectors
    ex = np.zeros(np.shape(ez))
    ex[0] = (ez[0,1], -ez[0,0], 0) # compute the first ex vector
    ex[0] = ex[0] / np.linalg.norm(ex[0]) # normalize the first ex vector
    for i in range(centroids.shape[0] - 1):
        b = np.cross(ez[i], ez[i + 1]) # compute cross product of consecutive tangents
        b = b / np.linalg.norm(b) # normalize vectore
        phi = np.arccos(np.dot(ez[i], ez[i + 1])) # compute angle between consecutive tangent
        R = compute_rot_mat(b,phi) # compute 3D rotation matrix
        ex[i + 1] = np.dot(R, ex[i]) # rotate previous ex and save it as current ex
            
    # Calculate the second normal vector ey
    ey = np.array([np.cross(t, n) for (t, n) in zip(ez, ex)])

    return ex,ey,ez

######################################################
# PIV IMPLEMENTATION
######################################################

def rot_mat(ph,th,ps):
    # R = np.array([  [ np.cos(th)+ux**2*(1-np.cos(th)), ux*uy*(1-np.cos(th))-uz*np.sin(th), ux*uz*(1-np.cos(th))+uy*np.sin(th) ], 
    #                 [ uy*ux*(1-np.cos(th))+uz*np.sin(th), np.cos(th)+uy**2*(1-np.cos(th)), uy*uz*(1-np.cos(th))-ux*np.sin(th) ],
    #                 [ ux*uz*(1-np.cos(th))-uy*np.sin(th), uy*uz*(1-np.cos(th))+ux*np.sin(th), np.cos(th)+uz**2*(1-np.cos(th)) ] ])

    R = np.array( [ [ np.cos(th)*np.cos(ph), np.sin(ps)*np.sin(th)*np.cos(ph)-np.cos(ps)*np.sin(ph), np.cos(ps)*np.sin(th)*np.cos(ph)+np.sin(ps)*np.sin(ph) ], 
                    [ np.cos(th)*np.sin(ph), np.sin(ps)*np.sin(th)*np.sin(ph)+np.cos(ps)*np.cos(ph), np.cos(ps)*np.sin(th)*np.sin(ph)-np.sin(ps)*np.cos(ph) ],
                    [ -np.sin(th), np.sin(ps)*np.cos(th), np.cos(ps)*np.cos(th) ] ] )
    return R

def rotate_piv(data,rot_params,keys):

    times = list(set(data[:,keys.index('timepoint')]))
    times = np.array(times).astype(np.uint16)
    rot_all = []
    for t in times:
        rot_all.append(rot_mat(rot_params[t,1],rot_params[t,2],rot_params[t,3]))

    rot_now = rot_mat(0,0,0)
    for t in times:
        print('#'*20,t)
        rot_now = np.matmul(rot_now,rot_all[t])
        print(rot_now)
        for i in range(data.shape[0]):
            cell = data[i]
            if cell[keys.index('timepoint')]==t:
                index = np.array([(k=='X')or(k=='Y')or(k=='Z') for k in keys])
                old_pos = cell[index]
                new_pos = np.matmul(rot_now,old_pos)
                data[i,index] = new_pos
    return data



