import xml.etree.ElementTree as ET
import os, glob
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams.update({'font.size': 20})
# plt.style.use('dark_background')
rc('pdf', fonttype=42)

def xml2numpy(path, keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore'], filename='file.txt'):
    flist = glob.glob(os.path.join(path,'*.xml'))
    flist.sort()
    # flist = flist[::-1]

    data = [[] for k in keys]

    max_cell_id = -1
    max_lineage_id = -1
    for t, f in enumerate(flist):
        tree = ET.parse(f)
        root = tree.getroot()
        print('#'*20)
        print(t,f, 'n cells:',len(root))
        print('#'*20)
        if t>0:
            prev_old_ids = [data[keys.index('old_id')][i] for i in range(len(data[0])) if data[keys.index('timepoint')][i]==(t-1)]
            prev_cell_ids = [data[keys.index('cell_id')][i] for i in range(len(data[0])) if data[keys.index('timepoint')][i]==(t-1)]
            prev_lineage = [data[keys.index('lineage')][i] for i in range(len(data[0])) if data[keys.index('timepoint')][i]==(t-1)]
            prev_parent = [data[keys.index('parent_id')][i] for i in range(len(data[0])) if data[keys.index('timepoint')][i]==(t-1)]

        parents = [int(i.attrib['parent']) for i in root]
        for idx,el in enumerate(root):
            # print(idx)
            cell = [-1 for k in keys]
            cell[keys.index('timepoint')] = t
            pos = [p for p in el.attrib['m'].split(' ')[:-1]]
            for i in np.arange(len(pos)):
                try:
                    pos[i] = float(pos[i])
                except:
                    pos[i] = np.nan
            pos = np.array(pos)
            scale = np.array([float(s) for s in el.attrib['scale'].split(' ')[:-1]])
            cell[keys.index('X')] = (pos*scale)[0]
            cell[keys.index('Y')] = (pos*scale)[1]
            cell[keys.index('Z')] = (pos*scale)[2]
            cell[keys.index('splitScore')] = int(el.attrib['splitScore'])
            cell[keys.index('old_id')] = int(el.attrib['id'])
            cell[keys.index('old_parentid')] = int(el.attrib['parent'])
            # assign a unique cell_id
            parent = int(el.attrib['parent'])
            if parent == -1:
                # if it's a new track, create a new cell id and a new lineage
                max_cell_id += 1
                max_lineage_id += 1
                cell[keys.index('cell_id')] = max_cell_id
                cell[keys.index('lineage')] = max_lineage_id
                cell[keys.index('parent_id')] = parent
            else:
                if np.sum([i==parent for i in parents])==2:
                    # if it's a daughter cell, assign new unique cell_id and same lineage
                    max_cell_id += 1
                    cell[keys.index('cell_id')] = max_cell_id
                    cell[keys.index('parent_id')] = prev_cell_ids[prev_old_ids.index(parent)]
                    cell[keys.index('lineage')] = prev_lineage[prev_old_ids.index(parent)]
                elif np.sum([i==parent for i in parents])==1:
                    # if it's the continuation of a tracked cell, copy the cell_id of the previous timepoint
                    cell[keys.index('cell_id')] = prev_cell_ids[prev_old_ids.index(parent)]
                    cell[keys.index('parent_id')] = prev_parent[prev_old_ids.index(parent)]
                    cell[keys.index('lineage')] = prev_lineage[prev_old_ids.index(parent)]
                else:
                    print('ERROR!!!!')

            # print(cell)
            for idx, val in enumerate(cell):
                data[idx].append(val)
    data = np.transpose(np.array(data))

    np.savetxt(filename,data, fmt='%i %i %i %i %i %i %1.4f %1.4f %1.4f %i')
    return data

def set_white_plot(ax):
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

def cell_number(data, visualize=True, ax = None, color='blue', keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']):

    time = list(set(data[:,keys.index('timepoint')]))
    n_cells = []
    n_confidence = []
    for i in time:
        data_onetp = data[data[:,keys.index('timepoint')]==i,:]
        n_tot = data_onetp.shape[0]
        n_cells.append(n_tot)
        n_conf_onetp = []
        for conf in range(4):
            n = data_onetp[data_onetp[:,keys.index('splitScore')]==conf].shape[0]
            n = n/n_tot
            n_conf_onetp.append(n)
        n_confidence.append(n_conf_onetp)
    n_confidence = np.array(n_confidence)
    n_cells = np.array(n_cells)

    if visualize:
        if not ax:
            fig, ax = plt.subplots(1,1,figsize=(8,4))
        ax.plot(time, n_cells,'-',color=color,lw=4)
    
    return (n_cells, n_confidence)

def cell_quality(data, visualize=True, ax = None, color='blue'):

    n_cells = []
    n_confidence = []
    for i in set(data[:,keys.index('timepoint')]):
        data_onetp = data[data[:,keys.index('timepoint')]==i,:]
        n_tot = data_onetp.shape[0]
        n_cells.append(n_tot)
        n_conf_onetp = []
        for conf in range(4):
            n = data_onetp[data_onetp[:,keys.index('splitScore')]==conf].shape[0]
            n = n/n_tot
            n_conf_onetp.append(n)
        n_confidence.append(n_conf_onetp)
    n_confidence = np.array(n_confidence)
    n_cells = np.array(n_cells)

    if visualize:
        if not ax:
            fig, ax = plt.subplots(1,1,figsize=(8,4))
        ax.bar(range(len(n_cells)),n_confidence[:,0],bottom=0,width=1,facecolor='red',alpha=.5)
        ax.bar(range(len(n_cells)),n_confidence[:,1],bottom=n_confidence[:,0],width=1)
        ax.bar(range(len(n_cells)),n_confidence[:,2],bottom=n_confidence[:,0]+n_confidence[:,1],width=1)
        ax.bar(range(len(n_cells)),n_confidence[:,3],bottom=n_confidence[:,0]+n_confidence[:,1]+n_confidence[:,2],width=1,facecolor='green',alpha=.5)
        # plt.show()
    
    return (n_cells, n_confidence)

def division_rate(data):

    return

def track(data):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cell_ids = list(set(data[:,keys.index('cell_id')]))
    print(len(cell_ids))
    for cell in cell_ids[::10]:
        cell_track = np.array([ data[data[:,keys.index('cell_id')]==cell,keys.index('X')],
                    data[data[:,keys.index('cell_id')]==cell,keys.index('Y')],
                    data[data[:,keys.index('cell_id')]==cell,keys.index('Z')] ]).transpose()
        # print(cell, cell_track.shape)

        if cell_track.shape[0]>10:
            ax.plot(cell_track[:,0],cell_track[:,1],cell_track[:,2])
    plt.show()
        
    return

def track_animation(data):
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as p3
    import matplotlib.animation as animation

    # Fixing random state for reproducibility
    np.random.seed(19680801)


    def update_lines(num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
        return lines

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Fifty lines of random 3-D lines
    cell_ids = list(set(data[:,keys.index('cell_id')]))
    cell_tracks = []
    print(len(cell_ids))
    for cell in cell_ids[::10]:
        cell_track = np.array([ data[data[:,keys.index('cell_id')]==cell,keys.index('X')],
                    data[data[:,keys.index('cell_id')]==cell,keys.index('Y')],
                    data[data[:,keys.index('cell_id')]==cell,keys.index('Z')] ]).transpose()
        # print(cell, cell_track.shape)

        if cell_track.shape[0]>10:
            cell_tracks.append(cell_track.transpose())

    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in cell_tracks]

    # Setting the axes properties
    # ax.set_xlim3d([0.0, 1.0])
    # ax.set_xlabel('X')

    # ax.set_ylim3d([0.0, 1.0])
    # ax.set_ylabel('Y')

    # ax.set_zlim3d([0.0, 1.0])
    # ax.set_zlabel('Z')

    # ax.set_title('3D Test')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(cell_tracks, lines),
                                    interval=50, blit=False)

    plt.show()
        
    return

def track_length(data, ax=None,color='blue',keys = ['old_id','old_parentid','cell_id','parent_id','lineage','timepoint','X','Y','Z','splitScore']):
    from mpl_toolkits.mplot3d import Axes3D

    if not ax:
        fig, ax = plt.subplots(1,1)
    lineage_ids = list(set(data[:,keys.index('lineage')]))
    lengths = []
    for lineage in lineage_ids[::10]:
        length = len(list(set(data[data[:,keys.index('lineage')]==lineage,keys.index('timepoint')])))
        # print(cell, cell_track.shape)

        lengths.append(length)
    # counts, bins = np.histogram(lengths, bins=10)
    # ax.hist(bins[:-1], bins, weights=counts/np.sum(counts))
    ax.hist(lengths,range=(0,200),bins=50,alpha=.7,facecolor=color)
    ax.set_ylim(0,300)
    # plt.show()
        
    return

def radial_distribution(data, cm=None, ax=None,color='blue'):

    data = data[data[:,keys.index('timepoint')]==15,:]
    if not cm:
        cm = np.array([np.mean(data[:,keys.index('X')]),np.mean(data[:,keys.index('Y')]),np.mean(data[:,keys.index('Z')])])
    cm = np.array(cm)
    dists = []
    for c in data:
        pos = np.array([c[keys.index('X')],c[keys.index('Y')],c[keys.index('Z')]])
        dists.append(np.sqrt(np.sum(((cm-pos)*0.6)**2)))
    
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.hist(dists,range=(0,450),bins=50,alpha=.7,facecolor=color)
    ax.set_xlim(0,450)
    ax.set_ylim(0,500)
    # plt.show()
    return
