import xml.etree.ElementTree as ET
import os, glob
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import rc
from tifffile import imread, imsave
from matplotlib import gridspec
plt.rcParams.update({'font.size': 15})
# plt.style.use('dark_background')
rc('pdf', fonttype=42)

### CELL NUMBER OVER TIME
fig = plt.figure(figsize=(4,6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
ax = [0,0]
ax[0] = plt.subplot(gs[0])
ax[1] = plt.subplot(gs[1])
for xlabel_i in ax[0].get_xticklabels():
    xlabel_i.set_visible(False)
for xlabel_i in ax[0].get_yticklabels():
    xlabel_i.set_visible(False)
for xlabel_i in ax[0].get_xticklines():
    xlabel_i.set_visible(False)
for xlabel_i in ax[0].get_yticklines():
    xlabel_i.set_visible(False)
for b in ax:
    b.spines['bottom'].set_color('white')
    b.spines['top'].set_color('white')
    b.spines['right'].set_color('white')
    b.spines['left'].set_color('white')
    b.xaxis.label.set_color('white')
    b.yaxis.label.set_color('white')
    b.tick_params(axis='x', colors='white')
    b.tick_params(axis='y', colors='white')
ax[1].set_yticks([])
ax[1].set_xticks([0,500,1000])
ax[1].set_xlim(0,1318)
ax[1].set_xlabel('Length (pxl)')

###

_slice = 120
row = 530
img1 = imread('..\\raw\\pescoid1--C00--T00000.tif')[_slice]
img2 = imread('..\\restoredScaled\\T00000.tif')[_slice]
print(img1.shape)

img_show = np.zeros(img1.shape)
N = 6
list1 = (2*np.arange(N/2)).astype(np.uint8)
list2 = (2*np.arange(N/2)+1).astype(np.uint8)
for i in list1:
    img_show[:,i*int(img_show.shape[1]/N):(i+1)*int(img_show.shape[1]/N)] = img1[:,i*int(img_show.shape[1]/N):(i+1)*int(img_show.shape[1]/N)]
for i in list2:
    img_show[:,i*int(img_show.shape[1]/N):(i+1)*int(img_show.shape[1]/N)] = img2[:,i*int(img_show.shape[1]/N):(i+1)*int(img_show.shape[1]/N)]

ax[0].imshow(img_show, cmap='gray',clim=(0,1000))
ax[0].plot([100,img_show.shape[1]-100],[row,row],'--w',lw=2)

p1 = img1[row,100:int(img1.shape[1]-100)]
p2 = img2[row,100:int(img2.shape[1]-100)]
ax[1].plot(np.arange(len(p1))+100,p1/np.max(p1),lw=2,color='tab:blue',alpha=.7)
ax[1].plot(np.arange(len(p2))+100,p2/np.max(p2),lw=2,color='tab:orange',alpha=.7)

# fig.savefig('C:\\Users\\nicol\\Dropbox\\Postdoc\\Presentation\\20190918_AndorAcademy\\figures\\pescoid_stats\\plot_profile.pdf',dpi=300)

plt.show()
