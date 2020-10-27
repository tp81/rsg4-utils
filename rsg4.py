import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import skimage.filters
from skimage import data
import os
from glob import glob
import tifffile
from itertools import cycle
from numba import jit

def findshift(a,b,smooth_sigma=0.5):
    '''Find traslation of b wrt. a using FFT.
    
    a and b are considered to be square matrices.
    
    '''
    
    # Smooth input images
    a = skimage.filters.gaussian(a,smooth_sigma)
    b = skimage.filters.gaussian(b,smooth_sigma)
    
    # Calculate 2D Fast Fourier Transform
    af = np.fft.fft2(a)
    bf = np.fft.fft2(b)
    
    # Calculate cross-correlation between a and b
    abf = af*np.conj(bf)
    iabf = np.fft.fftshift(abs(np.fft.ifft2(abf)))
    
    # Find maxima of cross-correlation
    dn = np.argmax(iabf)
    di,dj = np.unravel_index(dn,abf.shape)
    ddi = iabf.shape[0]/2-di
    ddj = iabf.shape[1]/2-dj
    
    return ddi,ddj

def find_stripe_shift(stripe1,stripe2,overlap=0.1):
    '''Find traslation of stripe2 with respect to stripe1.'''
    
    margin_size = int(np.round(stripe1.shape[1]*overlap))
    margin1 = stripe1[:,-margin_size:]
    margin2 = stripe2[:,:margin_size]
    
    di, dj = findshift(margin1,margin2)
    
    return -di, stripe1.shape[1] - margin_size - dj

def find_stripe_shifts(stripes, overlap=0.1):
    dijs = [(0,0)]
    dij = (0,0)
    
    for s1, s2 in zip(stripes[:-1],stripes[1:]):
        di, dj = find_stripe_shift(s1,s2,overlap)
        
        dij = ( dij[0]+di, dij[1]+dj )
        
        dijs.append(dij)
        
    return dijs

def bkg_sub(stripes, debug=False):
    ss = np.stack(stripes).sum(axis=0)
    
    bkg_x = np.linspace(0,ss.shape[1],ss.shape[1])
    bkg_y = ss.min(axis=0)
    bkg_coeff = np.polyfit(bkg_x,bkg_y,2)
    bkg_yhat = np.polyval(bkg_coeff,bkg_x)
    
    if debug:
        plt.plot(bkg_x,bkg_y,'b.')
        plt.plot(bkg_x,bkg_yhat,'r-')
    
    stripe_bkg = np.repeat(bkg_yhat[np.newaxis,:],ss.shape[0],0)
    
    stripes_bkgc = [ s.astype(float)/stripe_bkg for s in stripes ]
    
    return stripes_bkgc

def napari_mosaic_line(stripes,dijs):
    v2 = napari.Viewer()
    
    for s, dij, cm in zip(stripes, dijs, cycle(['cyan','magenta'])):
        sl = v2.add_image(s, translate=dij)
        sl.contrast_limits = (0,s.max()/20)
        sl.opacity = 0.5
        sl.blending = 'additive'
        sl.colormap = cm

def make_mosaic(stripes, dijs=None, overlap=0.1):
    if dijs is None:
        dijs = find_stripe_shifts(stripes, overlap)
    
    sh = stripes[0].shape

    # Get dis (row) and djs (cols)
    dis = [ di for di, dj in dijs ]
    djs = [ dj for di, dj in dijs ]
    
    # Find bounds wrt stripes[0]
    mini = min(dis)
    maxi = max(dis) + sh[0]
    minj = min(djs)
    maxj = max(djs) + sh[1]
    
    #print(mini,maxi,minj,maxj)
    
    nsh = (int(maxi-mini+1),int(maxj-minj+1))
    #print(nsh)
    
    newshape = np.zeros(shape=nsh,dtype=stripes[0].dtype)
    #print(newshape.shape)
        
    for stripe, dij in zip(stripes,dijs):
        i0 = int(dij[0]-mini)
        j0 = int(dij[1]-minj)
        
        #print(i0,j0)
        
        newshape[i0:i0+sh[0],j0:j0+sh[1]] = stripe
    
    return newshape

def Reconstruct(indir, overlap=0.1, do_bkgsub=True):
    tiffs = glob(os.path.join(indir,'*.tif'))
    
    stripes = [tifffile.imread(im) for im in sorted(tiffs)]
    
    if do_bkgsub:
        stripes = bkg_sub(stripes)
    
    dijs = find_stripe_shifts(stripes, overlap=overlap)
    
    m1 = make_mosaic(stripes, dijs)
    
    return m1

