from skimage import measure
import numpy as np
#import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import sklearn
import skimage
from skimage.feature import local_binary_pattern
import scipy.io as sio
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import os
import SimpleITK as sitk
from auxiliary_func import *
import warnings
import siamxt
import sys

warnings.filterwarnings("ignore")
t1file = sys.argv[1]
flairfile = sys.argv[2]
outdir = sys.argv[3]
## load classifier ##
clf = joblib.load('classWMHchallenge17.pkl')

# parameters inicialization
radius =5
n_points = 5* radius # LBP 
METHOD = 'nri_uniform'
s_rs = 200 # resize
th = 0.7 # prob classifier threshold
data = []
    
## read data ##
T1 = sitk.ReadImage(t1file)
FLAIR = sitk.ReadImage(flairfile)
T1_data = sitk.GetArrayFromImage(T1)
FLAIR_data = sitk.GetArrayFromImage(FLAIR)
brain_data = wmseg(os.path.join(t1file))>0

## new auxiliary variables ##
T1_pre = np.zeros((T1_data.shape[0],s_rs,s_rs))
FLAIR_pre = np.zeros((T1_data.shape[0],s_rs,s_rs))
brain_pre = np.zeros((T1_data.shape[0],s_rs,s_rs))>0
final_wml_pred = np.zeros_like(brain_pre)
result = np.zeros_like(FLAIR_data)
  
## pre-processing per slice ##
T1_data = ianormalize(T1_data).astype(np.uint8)
FLAIR_data = ianormalize(FLAIR_data).astype(np.uint8)
brain_data = brain_data.astype(np.uint8)
ind = np.argwhere(T1_data>10)

for slice in np.arange(T1_data.shape[0]):
        t1_slice = T1_data[slice].copy()
        flair_slice = FLAIR_data[slice].copy()
        brain_slice = brain_data[slice].copy()
        if brain_slice.sum()>0: # if there is a brain, this 2D slice is going to be proceed
                T1_pre[slice] = ianormalize(scipy.misc.imresize(t1_slice[min(ind[:,1]):max(ind[:,1]),min(ind[:,2]):max(ind[:,2])], (s_rs,s_rs),interp='bicubic')) 
                FLAIR_pre[slice] = ianormalize(scipy.misc.imresize(flair_slice[min(ind[:,1]):max(ind[:,1]),min(ind[:,2]):max(ind[:,2])], (s_rs,s_rs),interp='bicubic'))
                brain_pre[slice] = scipy.misc.imresize(brain_slice[min(ind[:,1]):max(ind[:,1]),min(ind[:,2]):max(ind[:,2])], (s_rs,s_rs),interp='bicubic')>0
            
## compute texture attributes ##
for slice in np.arange(T1_data.shape[0]):   
        flair_slice = FLAIR_pre[slice]
        brain_slice = brain_pre[slice]
        if brain_slice.sum()>0:
                #wm segmentation
                wmmean = flair_slice[brain_slice].mean()
                wm = brain_slice.copy()
                #compute image texture descriptors
                lbp_data = local_binary_pattern(flair_slice, n_points, radius, METHOD)
                grad_data = gradimg(flair_slice)
                mgrad_data = iagradm(flair_slice)
                lbp_data = 1.0*lbp_data/lbp_data.max()
                grad_data = 1.0*grad_data/grad_data.max()
                mgrad_data = 1.0*mgrad_data/mgrad_data.max()
                # ATT per pixel within WM with intensity> wmmean
                ind_class = argwhere(wm == 1)
                for ind_mask in xrange(len(ind_class)):
                        x = ind_class[ind_mask,0]
                        y = ind_class[ind_mask,1]
                        if slice < T1_data.shape[0]-1:
                                data.append([slice,x,y,T1_pre[slice,x,y],flair_slice[x,y],FLAIR_pre[slice-1,x,y],FLAIR_pre[slice+1,x,y],lbp_data[x,y],grad_data[x,y],mgrad_data[x,y],wmmean]) #
                        else:
                                data.append([slice,x,y,T1_pre[slice,x,y],flair_slice[x,y],FLAIR_pre[slice-1,x,y],FLAIR_pre[slice,x,y],lbp_data[x,y],grad_data[x,y],mgrad_data[x,y],wmmean])
 
## testing patient - Accuracy rate ##
data = array(data) 
Xtest = data[:,3::] # [slice,x,y,intensity,lbp,grad,..]
ypred = clf.predict_proba(Xtest)#clf.predict(Xtest)
ypred = (ypred[:,1]>th).astype(int)
    
## testing patient - predict 2D WMH segmentation on the data space ##
for slice in np.arange(T1_data.shape[0]):
        brain_slice = brain_pre[slice].copy()
        if brain_slice.sum()>0: # if there is a brain, this slice is going to be proceed
                mask_pred = np.zeros_like(brain_slice)
                indices = np.argwhere(data[:,0].astype(np.uint8)==slice) # indices from 
                if len(indices)!=0:
                        for ind_pixel in indices:
                                if int(ypred[ind_pixel]) == 1: # if this sample was classified as a WML
                                        info = data[ind_pixel].astype('int')[0] # take this sample information (image,slice,x,y,...) 
                                        mask_pred[info[1],info[2]] = 1 # set 1 (wml) to that corresponding pixel in the predicted (result) mask
                        # post-process #
                        mask_pred = iaareaopen(mask_pred.astype(int),10)#10,5,20
                        final_wml_pred[slice] = mask_pred.copy() 
## testing patient - 3D WMH segmentation on the original patient space ##
for slice in np.arange(T1_data.shape[0]):
        brain_slice = brain_pre[slice].copy()
        if brain_slice.sum()>0: # if there is a brain, this slice is going to be proceed
                aux = scipy.misc.imresize(final_wml_pred[slice].astype(uint8), (max(ind[:,1]) - min(ind[:,1]),max(ind[:,2]) - min(ind[:,2])),interp='bicubic')>0
                result[slice,min(ind[:,1]):max(ind[:,1]),min(ind[:,2]):max(ind[:,2])] = aux.copy()

output = sitk.GetImageFromArray(result.astype(np.uint8).copy())
output.SetSpacing( FLAIR.GetSpacing() )
output.SetOrigin( FLAIR.GetOrigin() )
output.SetDirection( FLAIR.GetDirection() )
sitk.WriteImage( output, os.path.join(outdir+'/result.nii.gz') )


