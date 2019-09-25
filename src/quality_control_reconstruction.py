import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
import os

path = os.path.join('..','..','..','Kerim_Anlas','Pescoid_SPIM_12_02_19','2019-02-12_16.41.38','111_sigmoid_noc')

img1 = imread(os.path.join(path,'pescoid1--C00--T00037.tif')).flatten()
percs = np.percentile(img1,(0.3,99.7))
img1 = np.clip(img1,percs[0],percs[1])
img1 = ((2**16-1)*(img1-np.min(img1))/(np.max(img1)-np.min(img1))).astype(np.uint16)
img2 = imread(os.path.join(path,'restoredFull','pescoid1--C00REC--T00037.tif')).flatten()

img1 = img1[::10000]
img2 = img2[::10000]

plt.plot(img1,img2,'o',alpha=.4)
plt.show()
