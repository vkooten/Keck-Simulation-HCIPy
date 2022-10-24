#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:54:37 2021

@author: maaikevankooten and Charlotte Bond
"""

from hcipy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import copy
import scipy.ndimage as ndimage
import datetime
# For notebook animations
from matplotlib import animation
from IPython.display import HTML
import numpy as np
import sys
from astropy.io import fits

# Comppute stellar flux
def flux(mag,band,wfs_tp,dt):
    # Need to check this calculation
    if band=='H':
        f0=93.3*0.29*10**8.*(73.4) #photons/m**2/s, mag 0 star, H band; from http://www.astronomy.ohio-state.edu/~martini/usefuldata.html using f0 = f_lambda*Delta(lambda), accounting for units
        flux_object=f0*10.**(-mag/2.5) # Flux 
        nPhotons=flux_object*dt*wfs_tp #photons
    else:
        print('"'+band+'" band not defined: select valid band')
        nPhotons = 1
    return nPhotons

def bin(imin,fbin): #bin PWFS images into 64x64
    out=np.zeros((int(imin.shape[0]/fbin),int(imin.shape[1]/fbin)))
    #begin binning
    for i in np.arange(fbin-1,imin.shape[0]-fbin,fbin):
        for j in np.arange(fbin-1,imin.shape[1]-fbin,fbin):
            out[int((i+1)/fbin)-1,int((j+1)/fbin)-1]=np.sum(imin[i-int((fbin-1)/2):i+int((fbin-1)/2),j-int((fbin-1)/2):j+int((fbin-1)/2)])
    return out

def pyramid_slopes(image,pwfs_grid,pwfs_mask):
    N=np.sum(pwfs_mask)/4
    norm=np.sum(image[pwfs_mask[:,0]]+image[pwfs_mask[:,1]]+image[pwfs_mask[:,2]]+image[pwfs_mask[:,3]])/N
    sx=(image[pwfs_mask[:,0]]-image[pwfs_mask[:,1]]+image[pwfs_mask[:,2]]-image[pwfs_mask[:,3]])/norm
    sy=(image[pwfs_mask[:,0]]+image[pwfs_mask[:,1]]-image[pwfs_mask[:,2]]-image[pwfs_mask[:,3]])/norm
    return np.array([sx,sy]).flatten()

def slopeMaps(slopes,pupil_mask):

    sx=np.zeros(pupil_mask.shape)
    sy=np.zeros(pupil_mask.shape)

    mid = int(slopes.size/2)
    sx[pupil_mask]=slopes[0:mid]
    sy[pupil_mask]=slopes[mid::]

    return [sx,sy]

def make_keck_aperture(normalized=True, with_spiders=False, with_segment_gaps=False, gap_padding=0, segment_transmissions=1, return_header=False, return_segments=False):
    pupil_diameter = 10.95 #m actual circumscribed diameter
    actual_segment_flat_diameter = np.sqrt(3)/2 * 1.8 #m actual segment flat-to-flat diameter
    # iris_ao_segment = np.sqrt(3)/2 * .7 mm (~.606 mm)
    actual_segment_gap = 0.003 #m actual gap size between segments
    # (3.5 - (3 D + 4 S)/6 = iris_ao segment gap (~7.4e-17)
    spider_width = 1*2.6e-2#Value from Sam; Jules value: 0.02450 #m actual strut size
    if normalized: 
        actual_segment_flat_diameter/=pupil_diameter
        actual_segment_gap/=pupil_diameter
        spider_width/=pupil_diameter
        pupil_diameter/=pupil_diameter
    gap_padding = 1.
    segment_gap = actual_segment_gap * gap_padding #padding out the segmentation gaps so they are visible and not sub-pixel
    segment_transmissions = 1.

    segment_flat_diameter = actual_segment_flat_diameter - (segment_gap - actual_segment_gap)
    segment_circum_diameter = 2 / np.sqrt(3) * segment_flat_diameter #segment circumscribed diameter

    num_rings = 3 #number of full rings of hexagons around central segment

    segment_positions = make_hexagonal_grid(actual_segment_flat_diameter + actual_segment_gap, num_rings)
    segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

    segment = hexagonal_aperture(segment_circum_diameter, np.pi / 2)

    spider1 = make_spider_infinite([0, 0], 0, spider_width)
    spider2 = make_spider_infinite([0, 0], 60, spider_width)
    spider3 = make_spider_infinite([0, 0], 120, spider_width)
    spider4 = make_spider_infinite([0, 0], 180, spider_width)
    spider5 = make_spider_infinite([0, 0], 240, spider_width)
    spider6 = make_spider_infinite([0, 0], 300, spider_width)

    segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments=True)

    segmentation, segments = segmented_aperture
    def segment_with_spider(segment):
        return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
    segments = [segment_with_spider(s) for s in segments]
    contour = make_segmented_aperture(segment, segment_positions)

    def func(grid):
        res = contour(grid) * spider1(grid) * spider2(grid) * spider3(grid)* spider4(grid) * spider3(grid)* spider5(grid) * spider6(grid) # * coro(grid)
        return Field(res, grid)
    if return_segments:
        return func, segments
    else:
        return func

# Define function from rad of phase to m OPD
def aber_to_opd(aber_rad, wavelength):
    aber_m = aber_rad * wavelength / (2 * np.pi)
    return aber_m

# Setup the pyramid wfs
def setupPyWFS(grid,mod,wavelength,D,modSteps):
    mod_r = mod*wavelength/D    # modulation radius in radians
    pwfs = PyramidWavefrontSensorOptics(grid, wavelength_0=wavelength)
    mpwfs = ModulatedPyramidWavefrontSensorOptics(pwfs,mod_r,modSteps)   
    wfs_camera = NoiselessDetector(grid)
    return pwfs,mpwfs,wfs_camera

# Propagate the PyWfs
# Propagates a given wavefront to the wfs and integrates through the modulation
# circle. The detector is binned and the slopes computed
def propagatePyWFS(wfs_camera,mpwfs,wf,modSteps,bin_wfs,pwfs_grid,pwfs_mask,pwfs_refSlopes,dt):
    for m in range (modSteps) :
        if np.size(wf)==1:
            wfs_camera.integrate(mpwfs(wf)[m],1)
        else:
            wfs_camera.integrate(wf[m],dt/modSteps)

    pwfs_im = bin(wfs_camera.read_out().shaped,bin_wfs).flatten()
    pwfs_slopes = pyramid_slopes(pwfs_im/pwfs_im.sum(),pwfs_grid,pwfs_mask) - pwfs_refSlopes

    return pwfs_im, pwfs_slopes

# Initiate PyWFS
# Propagates the wfs and computes the mask and reference slopes/image
def initPyWFS(wfs_camera,mpwfs,wf,modSteps,bin_wfs,pwfs_grid,wfs_thres,dt):
    pwfs_mask = np.ones([pwfs_grid.size,4])==1
    pwfs_im, pwfs_slopes = propagatePyWFS(wfs_camera,mpwfs,wf,modSteps,bin_wfs,pwfs_grid,pwfs_mask,0,dt)
    # Sum the four pupil images (assumes they are at equivalent pixel locations in each quadrant)
    im_sq = pwfs_im.reshape(pwfs_grid.shape)
    N = int(pwfs_grid.shape[0]/2)
    im_pup = im_sq[:N,:N]+im_sq[:N,N:]+im_sq[N:,:N]+im_sq[N:,N:]
    # Compute a thresholded mask
    im_mask = np.zeros([N,N])
    im_mask[im_pup>(wfs_thres*np.max(im_pup))] = 1
    pupil_mask = im_mask==1
    pwfs_mask = np.zeros([2*N,2*N,4])
    pwfs_mask[:N,:N,0]=im_mask
    pwfs_mask[N:,:N,1]=im_mask
    pwfs_mask[:N,N:,2]=im_mask
    pwfs_mask[N:,N:,3]=im_mask
    pwfs_mask = pwfs_mask.reshape([pwfs_grid.size,4])==1
    # Re-propagate
    pwfs_im, pwfs_refSlopes = propagatePyWFS(wfs_camera,mpwfs,wf,modSteps,bin_wfs,pwfs_grid,pwfs_mask,0,dt)

    return pwfs_im, pwfs_refSlopes, pwfs_mask, pupil_mask

# Setup the DM
def setupZonalDM(nAct,D,grid,crosstalk):
    d = D/nAct     # Why D_full not D?
    inf = make_gaussian_influence_functions(grid,nAct,d,crosstalk=crosstalk)
    dm = DeformableMirror(inf)
    nModes = dm.num_actuators
    ap=circular_aperture(D)
    dm_pupil=ap(make_pupil_grid(nAct,D))
    return dm, nModes, dm_pupil

# Calibrate the AO system
def calibratePyWFS(dm,validModes,pupil,mpwfs,wfs_camera,wavelength,modSteps,bin_wfs,pwfs_grid,pwfs_mask,pwfs_refSlopes,probe_amp,rcond,dt):

    RM = []

    wf = Wavefront(pupil,wavelength)
    wf.total_power = 1

    # Code to generate command matrix
    nModes = np.sum(validModes)
    print('Calibrating %d modes...'%(nModes))
    amp = np.zeros((int(nModes),))
    for i in range(int(nModes)):
        print('Calibrating mode %d/%d'%(i+1,nModes),end='\r')
        slope = 0
    
        # Probe the phase response
        for s in [1, -1]:
            amp[:] = 0
            amp[i] = s * probe_amp
            dm.flatten()
            dm.actuators[validModes==1] = amp
            
            dm_wf = dm.forward(wf)
            wfs_wf = mpwfs.forward(dm_wf)

            # Propagate PyWFS
            pwfs_im, pwfs_slopes = propagatePyWFS(wfs_camera,mpwfs,wfs_wf,modSteps,bin_wfs,pwfs_grid,pwfs_mask,pwfs_refSlopes,dt)

            slope += s * (pwfs_slopes)/(2*probe_amp)

        RM.append(slope.ravel())

    IM_basis= ModeBasis(RM)
    # Compute the SVD
    svd = SVD(IM_basis.transformation_matrix)
    CM = computeCM(IM_basis.transformation_matrix,rcond,svd)

    return IM_basis.transformation_matrix, CM, svd

def computeCM(IM,rcond,svd=None):
    if svd is None:
        svd = SVD(M)

    CM = inverse_truncated(IM,rcond,svd)

    # Plot eigenvalues
    plt.semilogy(svd.S/svd.S.max())
    plt.semilogy(rcond*np.ones(np.size(svd.S)))
    plt.xlabel('Mode #')
    plt.ylabel('Eigenvalue')
    plt.show()
    
    return CM
    
def recomputeCM(IM,nModes,rcond,mode_dm_proj):
    nModes_old = np.shape(IM)[1]
    if nModes_old<nModes:
        print('WARNING: higher # modes requested than calibrated!')
        print('Using %g modes only'%(nModes_old))
        IM_new = IM
    else:
        IM_new = IM[:,0:nModes]
        
    svd = SVD(IM_new)
    CM_modal = computeCM(IM_new,rcond,svd)
    CM = np.matmul(mode_dm_proj[:,0:nModes],CM_modal)
    return CM
    
# Function to apply modal gains to the command matrix
# 
def applyModalGains(cm,cutOffFreq,nAct,dm_pupil,mGains):

    # Compute required modal loop gains.  Eventually this will be
    # replaced with estimatecompensation of estimated  modal optical gains.

    # Number of groups of modes.  Start with 2, eventually will
    # compute gain for each mode
    nGroups = np.size(mGains)

    # Apply gains
    # For 2 groups apply one set of gains in slope region
    # and one set in flat region (theoretical sensitivity)
    # Number of mdoes (- piston)
    nModes = nAct*nAct

    # Compute modes (DM projection)
    [fmodes_full,l,k] =  fourierModes(nAct,nAct)

    # Take just the valid actuators
    pup = dm_pupil==1
    fmodes = fmodes_full[:,pup]

    # Compute radial spatial frequency
    fr = np.sqrt(l*l+k*k)
    
    # Modal gain vector (no piston)
    modalGains = np.zeros(nModes)
    if nGroups==nModes:
        modalGains = mGains
    elif nGroups==2:
        # 2 groups
        # Cut-off frequency from modulation
        i_high = fr>cutOffFreq
        modalGains[~i_high] = mGains[0]
        modalGains[i_high] = mGains[1]
    else:
        # For now do nothing (equal gains) if no. groups
        # is not all or 2
        modalGains[:] = 1

    # Set piston gain to 0
    modalGains[fr==0] = 0

    plt.imshow(np.reshape(modalGains,[nAct,nAct]))
    plt.show()
        
    # Apply modal gains to reconstructor
    # Inverse (for projection)
    ifmodes = np.linalg.pinv(fmodes)
    # Fourier reconstructor
    cm_f = np.matmul(np.transpose(cm),ifmodes)

    cLim = np.max(np.abs(cm_f))
    plt.imshow(cm_f,aspect='auto',cmap='seismic',vmin=-cLim,vmax=cLim)
    plt.colorbar()
    plt.show()
    
    print(np.shape(cm_f))
    for n in range(0,nModes):
        cm_f[:,n] = modalGains[n]*cm_f[:,n]

    cLim = np.max(np.abs(cm_f))
    plt.imshow(cm_f,aspect='auto',cmap='seismic',vmin=-cLim,vmax=cLim)
    plt.colorbar()
    plt.show()
        
    # New zonal cm
    cm_new = np.matmul(np.transpose(fmodes),np.transpose(cm_f))

    cLim = np.max(np.abs(cm_new))
    plt.imshow(cm_new,aspect='auto',cmap='seismic',vmin=-cLim,vmax=cLim)
    plt.colorbar()
    plt.show()

    # Plot identity
    
    
    return cm_new


# Function to compute Fourier modes (commands) for DM.  Computes
# all fourier modes specificed by a given resolution actuator
# resolution.
#
# Doesn't include piston
#
# Inputs: nAct - Number of actuators across the pupil
#         nPx  - Resolution of modes (for the purposes of DM
#                commands nPx = nAct)

def fourierModes(nAct,nPx):

    # x/y values
    [n,m] = np.meshgrid(np.linspace(0,nPx-1,nPx),np.linspace(0,nPx-1,nPx))

    # Fourier modes - piston
    fmodes = np.zeros([nAct*nAct,nPx,nPx])

    # Spatial frequencies
    ll = np.linspace(-np.floor(nAct/2),np.floor((nAct-1)/2),nAct)
    kk = np.linspace(-np.floor(nAct/2),np.floor((nAct-1)/2),nAct)

    # Central frequency (i.e. [n0,n0] is the piston mode)
    n0 = np.ceil((nAct+1)/2)-1

    # Store spatial frequencies
    l = np.zeros(nAct*nAct)
    k = np.zeros(nAct*nAct)

    # Mode counting
    modeNb = 0

    # Compute all modes
    for i in range(0,nAct):
        for j in range(0,nAct):

            # Compute Fourier modes
            if nAct%2 == 0 and i==0 and j>n0:
                mode = np.sin(2*np.pi/nPx*(-ll[i]*n-kk[j]*m))
            elif i<n0 or (i==n0 and j<=n0):
                mode = np.cos(2*np.pi/nPx*(-ll[i]*n-kk[j]*m))
            else:
                mode = np.sin(2*np.pi/nPx*(-ll[i]*n-kk[j]*m))

            fmodes[modeNb,:,:] = mode
            l[modeNb] = ll[i]
            k[modeNb] = kk[j]
                
            modeNb = modeNb+1

    return fmodes, l, k
            

def EOF_filter(D,data,alpha=1,flag=0):
    '''
    D: the regressors matrix or the matrix of histroy vectors --> for Keck should be [(m*n),l] vector where m is number of modes, n is the temooral depth of the filter, l is the numer of training sets
    data: is the data vector containing the wavefront measurements delayed by the lag. The shape should be [(m*n),l]
    using the method from Jensen-Clem 2019 for the filter approximation
    '''
    #each wavefront point has a seperate filter so we need to calcualte this all. The regressors stay the same but the data that is give by the filter is differnt. 
    
    F=[]
    idenity=np.eye(D.shape[0])
  #  print(D.shape)
 #   print(data.shape)
 #   print(np.matmul(D,D.transpose()).shape)

    for i in range(data.shape[0]):
        if flag:
           temp1=inverse_truncated(D.transpose(), rcond=1e-4) #need to figet with the rounding condition
           f=np.matmul(temp1,data[i,:].transpose()).transpose()

        else:
            temp1=np.linalg.inv(np.matmul(D,D.transpose())+alpha*idenity)
       # print(temp1.shape)
            temp2=np.matmul(data[i,:],D.transpose())
       # print(temp2.shape)
            f=np.matmul(temp2,temp1)
        
        F.append(f)
#    print(temp1.shape)
#    print(temp2.shape)
#    print(f.shape)
    return np.array(F)


def setup_prediction():
    #everything in here will be place in a function to initialize
    temp_order=5
    l=2500 #needs to be much larger
    dm_pred, nModes, dm_pupil = setupZonalDM(nAct,D,pupil_grid,crosstalk)
    pwfs_pred, mpwfs_pred, wfs_camera_pred = setupPyWFS(pupil_grid,mod,lambda_wfs,D,modSteps)

    open_loop_phase=[]
    pred_rms=[]
    old_data=[]
    if pred_type=='EOF':      
        a=1. #alpha value
        spatial_order=349 #number of actuators but can be changed
        regressors=[]
        data=[]
        for kkk in range(temp_order):
            old_data.append(np.zeros(int(spatial_order)))

    else:
        import data_handle
        import LMMSE_large     
        spatial_order=3 #number of spatial regressors in a nxn grid around point predicted
        size=21 #number of actuators
        data_feed=data_handle.data_handle(size,spatial_order,temp_order) 
        delay_feed=data_handle.data_handle(size,spatial_order,temp_order)

        if pred_type=='LMMSE-r':
            predictor=LMMSE_large.LMMSE(spatial_order,temp_order,size**2, t='recursive')#initialize
        else:
            predictor=LMMSE_large.LMMSE(spatial_order,temp_order,size**2,t='forgetting',forgetting=f)#initialize

        return dm_pred, nModes, dm_pupil, pwfs_pred, mpwfs_pred, wfs_camera_pred, predictor, data_feed, delay_feed, 

def make_keck_Lband_lyot(normalized=True, with_spiders=False, with_segment_gaps=False, gap_padding=0, segment_transmissions=1, return_header=False, return_segments=False):
    """
    

    Parameters
    ----------
    normalized : TYPE, optional
        DESCRIPTION. The default is True.
    with_spiders : TYPE, optional
        DESCRIPTION. The default is False.
    with_segment_gaps : TYPE, optional
        DESCRIPTION. The default is False.
    gap_padding : TYPE, optional
        DESCRIPTION. The default is 0.
    segment_transmissions : TYPE, optional
        DESCRIPTION. The default is 1.
    return_header : TYPE, optional
        DESCRIPTION. The default is False.
    return_segments : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    conversion=(10.95/(2*12.05/1000))
    pupil_diameter = 10.95 #m actual circumscribed diameter
    actual_segment_flat_diameter = np.sqrt(3)/2 *0.00337*conversion #m actual segment flat-to-flat diameter
    # iris_ao_segment = np.sqrt(3)/2 * .7 mm (~.606 mm)
    central_obscuration_diameter=0.00698*conversion
    actual_segment_gap = 0.003 #m actual gap size between segments
    # (3.5 - (3 D + 4 S)/6 = iris_ao segment gap (~7.4e-17)
    spider_width = 0.0005*conversion#Jules previous value: 0.02450 #m actual strut size
    if normalized: 
        actual_segment_flat_diameter/=pupil_diameter
        actual_segment_gap/=pupil_diameter
        spider_width/=pupil_diameter
        pupil_diameter/=pupil_diameter
    gap_padding = 10.
    segment_gap = actual_segment_gap * gap_padding #padding out the segmentation gaps so they are visible and not sub-pixel
    segment_transmissions = 1.

    segment_flat_diameter = actual_segment_flat_diameter - (segment_gap - actual_segment_gap)
    segment_circum_diameter = 2 / np.sqrt(3) * segment_flat_diameter #segment circumscribed diameter

    num_rings = 3 #number of full rings of hexagons around central segment

    segment_positions = make_hexagonal_grid(actual_segment_flat_diameter + actual_segment_gap, num_rings)
    segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

    segment = hexagonal_aperture(segment_circum_diameter, np.pi / 2)

    spider1 = make_spider_infinite([0, 0], 0, spider_width)
    spider2 = make_spider_infinite([0, 0], 60, spider_width)
    spider3 = make_spider_infinite([0, 0], 120, spider_width)
    spider4 = make_spider_infinite([0, 0], 180, spider_width)
    spider5 = make_spider_infinite([0, 0], 240, spider_width)
    spider6 = make_spider_infinite([0, 0], 300, spider_width)

    segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments=True)

    segmentation, segments = segmented_aperture
    def segment_with_spider(segment):
        return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
    segments = [segment_with_spider(s) for s in segments]
    contour = make_segmented_aperture(segment, segment_positions)

    def func(grid):
        ap=contour(grid)
        co=circular_aperture(central_obscuration_diameter)(grid)
        ap[co==1]=0
        res = (ap )* spider1(grid) * spider2(grid) * spider3(grid)* spider4(grid) * spider3(grid)* spider5(grid) * spider6(grid) # * coro(grid)
        return Field(res, grid)
    if return_segments:
        return func
    else:
        return func
