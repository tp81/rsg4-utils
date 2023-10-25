import numpy as np
import pandas as pd
import skimage
import skimage.filters
from skimage.transform import warp, SimilarityTransform
from skimage import data
import os
from glob import glob
import tifffile
from itertools import cycle

import cupyx
import cupyx.scipy.ndimage
import cupy

def findshift(a,b,smooth_sigma=0.5,use_gpu=True):
    '''Find traslation of b wrt. a using FFT.
    
    a and b are considered to be square matrices.
    
    '''
    
    if use_gpu:
        gaussian = cupyx.scipy.ndimage.gaussian_filter
        fft2 = cupy.fft.fft2
        ifft2 = cupy.fft.ifft2
        conj = cupy.conj
        fftshift = cupy.fft.fftshift
        argmax = cupy.argmax
        abs_ = cupy.abs

        a = cupy.array(a)
        b = cupy.array(b)
    else:
        gaussian = skimage.filters.gaussian
        fft2 = np.fft.fft2
        ifft2 = np.fft.ifft2
        conj = np.conj
        fftshift = np.fft.fftshift
        argmax = np.argmax
        abs_ = np.abs

    # Smooth input images
    a = gaussian(a,smooth_sigma)
    b = gaussian(b,smooth_sigma)
    
    # Calculate 2D Fast Fourier Transform
    af = fft2(a)
    bf = fft2(b)
    
    # Calculate cross-correlation between a and b
    abf = af*conj(bf)
    iabf = fftshift(abs_(ifft2(abf)))
    
    # Find maxima of cross-correlation
    dn = argmax(iabf)
    di,dj = np.unravel_index(dn,abf.shape)
    ddi = iabf.shape[0]/2-di
    ddj = iabf.shape[1]/2-dj
    
    return ddi,ddj

def find_stripe_shift(stripe1,stripe2,overlap=0.1,use_gpu=True):
    '''Find traslation of stripe2 with respect to stripe1.'''
    
    margin_size = int(np.round(stripe1.shape[1]*overlap))
    margin1 = stripe1[:,-margin_size:]
    margin2 = stripe2[:,:margin_size]
    
    di, dj = findshift(margin1,margin2,use_gpu)
    
    return -di, stripe1.shape[1] - margin_size - dj

def find_stripe_shifts(stripes, overlap=0.1,use_gpu=True):
    dijs = [(0,0)]
    dij = (0,0)
    
    for s1, s2 in zip(stripes[:-1],stripes[1:]):
        di, dj = find_stripe_shift(s1,s2,overlap,use_gpu)
        
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
        import matplotlib.pyplot as plt

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

def make_mosaic(stripes, dijs=None, overlap=0.1, use_gpu=True):
    if dijs is None:
        dijs = find_stripe_shifts(stripes, overlap, use_gpu)
    
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

def ReconstructLayer(indir, overlap=0.1, do_bkgsub=True, use_gpu=True):
    tiffs = glob(os.path.join(indir,'*.tif'))
    
    # Load all stripes in memory
    stripes = [tifffile.imread(im) for im in sorted(tiffs)]
    
    if do_bkgsub:
        stripes = bkg_sub(stripes)
    
    dijs = find_stripe_shifts(stripes, overlap=overlap)
    
    m1 = make_mosaic(stripes, dijs)
    
    return m1

def align_stack(images):
    n = len(images)

    aligned_images = [images[0]]
    reference_image = images[0]

    for i in range(1,n):
        cross_corr = findshift(reference_image, images[i])
        y, x = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        shift_y, shift_x = np.array(cross_corr.shape) / 2 - np.array([y, x])
        translation_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
        aligned_image = images[i].get()
        aligned_image = warp(aligned_image, SimilarityTransform(translation_matrix))
        aligned_images.append(aligned_image)
    
    # Stack the aligned images
    aligned_stack = np.stack(aligned_images, axis=0)

    return aligned_stack

def ReconstructStack(indir, outdir, overlap=0.1, do_bkgsub=True, use_gpu=True):
    # Get all the layers
    layersDirs = [p for p in glob(os.path.join(indir,'layer*')) if os.path.isdir(p) ]

    aligned_stack_images={}

    # In each layer, process each channel
    for ldir in layersDirs:
        for chdir in [ p for p in glob(os.path.join(ldir, '*')) if os.path.isdir(p) ]:
            chname = os.path.split(chdir)[-1]
            if not chname in aligned_stack.keys():
                aligned_stack_images[chname] = []

            # Stitch all layers
            ch = ReconstructLayer(os.path.join(chdir,'images'), 
                                 overlap=overlap, do_bkgsub=do_bkgsub, use_gpu=use_gpu)
            
            # Save output
            aligned_stack_images[chname].append(ch)

    aligned_stack={}
    for chname in aligned_stack_images.keys():
        aligned_stack[chname] = np.stack(aligned_stack_images[chname])

        outname = f'stack_{chname}.tif'

        outpath = os.path.join(outdir, outname)

        tifffile.imwrite(outpath, aligned_stack[chname])

