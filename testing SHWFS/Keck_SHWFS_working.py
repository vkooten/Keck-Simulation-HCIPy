#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:54:37 2021

@author: maaikevankooten
"""

from hcipy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import copy
import scipy.ndimage as ndimage

# For notebook animations
from matplotlib import animation
from IPython.display import HTML
from hcipy import *
import hcipy
import numpy as np
import matplotlib.pyplot as plt


def make_keck_aperture(normalized=True, with_spiders=False, with_segment_gaps=False, gap_padding=0, segment_transmissions=1, return_header=False, return_segments=False):
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

    pupil_diameter = 10.95 #m actual circumscribed diameter
    actual_segment_flat_diameter = np.sqrt(3)/2 * 1.8 #m actual segment flat-to-flat diameter
    # iris_ao_segment = np.sqrt(3)/2 * .7 mm (~.606 mm)
    central_obscuration_diameter=2.6
    actual_segment_gap = 0.01#0.003 #m actual gap size between segments
    # (3.5 - (3 D + 4 S)/6 = iris_ao segment gap (~7.4e-17)
    spider_width = 0.01#2*2.6e-2#Jules previous value: 0.02450 #m actual strut size
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
    #segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

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
        return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)* spider5(grid)* spider6(grid)
    def segment_with_secondary(segment):
        return lambda grid: segment(grid) * (1-circular_aperture(central_obscuration_diameter)(grid))* spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)* spider5(grid)* spider6(grid)
    segments = [segment_with_secondary(s) for s in segments]
    contour = make_segmented_aperture(segment, segment_positions)


    def func(grid):
        ap=contour(grid)
        co=circular_aperture(central_obscuration_diameter)(grid)
        ap[co==1]=0
        res = (ap )* spider1(grid) * spider2(grid) * spider3(grid)* spider4(grid) * spider5(grid) * spider6(grid) # * coro(grid)
        return Field(res, grid)
    if return_segments:
        return func, segments
    else:
        return func

# Define function from rad of phase to m OPD
def aber_to_opd(aber_rad, wavelength):
    aber_m = aber_rad * wavelength / (2 * np.pi)
    return aber_m
    ######need to fix these such that you feed the science images with the noise.
    S=0.6
    try:
        I2=wf_out.intensity
        I1=wf_in.intensity

    except:
        print('Recieved images')
        I2=np.sqrt(wf_out)*telescope_pupil/dt #this is for when I feed the science image.
        I1=np.sqrt(wf_in)*telescope_pupil/dt
  #  I2/=sum(I1**2)
   # I1/=sum(I1**2)
    # print(I2.sum())
    # print(I1.sum())
    wf_lf = Wavefront(telescope_pupil)
    wf_lf.electric_field = np.sqrt(I1)*telescope_pupil
    focal_grid = make_focal_grid(q=20, num_airy=2)#, f_number=5.65, reference_wavelength=my_wavelength)


    M = Apodizer(circular_aperture(zernike_diameter)(focal_grid))
    # Calculating b0
    prop = FraunhoferPropagator(zernike_pupil_grid, focal_grid)
    b0 = prop.backward(M(prop.forward(wf_lf))).electric_field.real
    I1lf = prop.backward(M(prop.forward(wf_lf))).intensity

    # Calculating b
    b = np.sqrt(S) * b0

    # N’Diaye et al. 2013
    phase_est = -1 + np.sqrt((3 - 2 * b - (1 - ((I2)*telescope_pupil)) / b))
    phase_est*=telescope_pupil
    # Wallace et al. in prep
    phase_wallace = np.pi/4 + np.arcsin((I2-I1-2*I1lf)  / (2*np.sqrt(I1*2*I1lf)))

    # Steeves et al. in prep
    P = np.sqrt(I1)
    phase_mz = np.pi/4 - np.arcsin( (P**2 + 2*b**2 - I2) / (2*np.sqrt(2)*P*b)  )
    return phase_est, phase_wallace, phase_mz

plt.close('all')
grid_size=1024
telescope_diameter=10.95
pupil_grid = make_pupil_grid(grid_size, telescope_diameter)

keck_aperture, segments = make_keck_aperture(return_segments=True,normalized=False)
telescope_pupil = evaluate_supersampled(keck_aperture, pupil_grid, 8)
segments = evaluate_supersampled(segments, pupil_grid, 8)
#keck_aperture  =make_keck_aperture(telescope_diameter)
#telescope_pupil= keck_aperture(pupil_grid)

#keck_aperture  =circular_aperture(telescope_diameter)
#telescope_pupil = evaluate_supersampled(keck_aperture, pupil_grid, 2)
wavelength_wfs = 1.65e-6 #SHWFS is different wavelength than th others
angular_scale=wavelength_wfs/telescope_diameter #l/D
k=2*np.pi/wavelength_wfs
###########setup the star & magnitude###################

stellar_magnitude=0#guide star magnitude at the given observing wavelength
f0=93.3*0.29*10**8. #photons/m**2/s, mag 0 star, H band; from http://www.astronomy.ohio-state.edu/~martini/usefuldata.html using f0 = f_lambda*Delta(lambda), accounting for units
#f0 = 3.9E10
dt=1/1200.
#flux_object_ini=f0*10.**(-stellar_magnitude/2.5)
#tr_atm,th,qe=0.9,0.2,0.8 #assume transmission through the atmosphere, instrument throughput, quantum efficiency of CCD
#flux_object=flux_object_ini#*tr_atm*th*qe
#Nphot=flux_object
#######setup the wavefront###################
wf_pup = Wavefront(telescope_pupil,wavelength=wavelength_wfs)
#wf_pup.total_power=1
wf_pup.total_power = f0 * 10**(-stellar_magnitude/2.5)
#wf_pup.Intensity = flux_object
#wf_pup.total_power = flux_object

#########setup the focal plane propagator##########
focal_grid = make_focal_grid(q=4, num_airy=20,spatial_resolution=angular_scale)
propagator = FraunhoferPropagator(pupil_grid, focal_grid)
Inorm = propagator.forward(wf_pup).power.max()
PSF_in = propagator.forward(wf_pup).power / Inorm
############## setup a simple ao residual code here but will need to furthur explore/improve to capture everything######
#doing it this way wont include the fact that the AO system will try to correct for the tip/tilt. will need full ao simulation for this I believe.
nm=500 #number of different 0.1 second sequences to average over; currently 500 simulated

fried_parameter =0.20 # meter
outer_scale = 50 # meter
vx,vy=5.,0. #windspeed in m/s
velocity = np.sqrt(vx**2.+vy**2.) # meter/sec
Cn_squared = Cn_squared_from_fried_parameter(fried_parameter)
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
#####use modal layer to get atmospheric residuals

f_number = 15 #NIRC2 on Keck II

#so the spot size is 2.44e-5
num_lenslets = 20 # 40 lenslets along one diameter
sh_diameter = 0.2e-3*(num_lenslets) # m -->or 4mm
#SHWFS_pupil=make_pupil_grid(16*20,diameter=sh_diameter)

magnification = sh_diameter /telescope_diameter
#magnification = 562.5/0.2

magnifier = Magnifier(magnification)

shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), f_number, num_lenslets, sh_diameter)
shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)

camera = NoiselessDetector(focal_grid)

camera.integrate(shwfs(magnifier(wf_pup)), dt)

image_ref = camera.read_out()


slopes_ref = shwfse.estimate([image_ref])
slopes_ref-=np.mean(slopes_ref)

num_actuators_across_pupil = 21
actuator_spacing = telescope_diameter / num_actuators_across_pupil
influence_functions = make_gaussian_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing)
deformable_mirror = DeformableMirror(influence_functions)
num_modes = deformable_mirror.num_actuators


probe_amp = 0.1 * wavelength_wfs
response_matrix = []

wf=wf_pup.copy()

plt.figure(figsize=(10, 6))

for i in range(int(num_modes)):
    slope = 0

    # Probe the phase response
    for s in [1, -1]:
        amp = np.zeros((num_modes,))
        amp[i] = s * probe_amp
        deformable_mirror.flatten()
        deformable_mirror.actuators = amp

        dm_wf = deformable_mirror.forward(wf)
        wfs_wf = shwfs(magnifier(dm_wf))

        camera.integrate(wfs_wf, dt)
        image = camera.read_out()

        slopes = shwfse.estimate([image+1e-15])
        slopes-=np.mean(slopes)

        slopes-=slopes_ref
        slope += s * (slopes.ravel())/(2*probe_amp)#np.var([-probe_amp,probe_amp])
       # slope += amp * slopes / np.var([-probe_amp,probe_amp])
        #slope += amp * slopes / 2

    response_matrix.append(slope.ravel())

    # if i > 40 and (i + 1) % 20 != 0:
    #     continue

    # # Plot mode response
    # plt.clf()
    # plt.suptitle('Mode %d / %d: DM shape' % (i + 1, num_modes))

    # plt.subplot(1,2,1)
    # plt.title('DM surface')
    # im1 = imshow_field(deformable_mirror.surface, cmap='RdBu')

    # plt.subplot(1,2,2)
    # plt.title('SH spots')
    # im2 = imshow_field(image,vmin=0,vmax=5e-5)
    # plt.pause(0.1)

response_matrix = ModeBasis(response_matrix)
rcond = 0.1

reconstruction_matrix = inverse_tikhonov(response_matrix.transformation_matrix, rcond=rcond)

##########SEGMENTED BASIS SET THAT CAN MOVE#####################
# Instantiate the segmented mirror
ksm =SegmentedDeformableMirror(segments)
 #lets get a reference image quickly
# Apply SM to pupil plane wf
wf=wf_pup.copy()
wf_sm = ksm.forward(wf_pup)

# Propagate from SM to image plane
im_ref_k= propagator.forward(wf_sm)
norm_k= np.max(im_ref_k.intensity)

ksm.flatten() #for future use lets flatten it quickly. Then plot what we had before
# Display intensity and phase in image plane
# plt.figure(figsize=(18, 9))
# plt.suptitle('Image plane after HCIPy SM')
# # Get normalization factor for HCIPy reference image
# plt.subplot(1, 2, 1)
# hcipy.imshow_field(np.log10(im_ref_k.intensity/norm_k))
# plt.title('Intensity')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# hcipy.imshow_field(im_ref_k.phase, cmap='RdBu')
# plt.title('Phase')
# plt.colorbar()


ksm.flatten()
wf=wf_pup.copy()
primary_wf=ksm.forward(wf)

plt.figure()
plt.subplot(1,2,1)
plt.title('Primary Mirror [radians]')
imshow_field(primary_wf.phase, cmap='RdBu',vmin=0,vmax=0.6)
plt.colorbar()

plt.subplot(1,2,2)
plt.title('WFS image [counts]')
imshow_field(image_ref, cmap='inferno',vmin=0,vmax=12000)
plt.colorbar()
plt.tight_layout()
plt.clf()

######apply an aberrated wavefront#############
# phase_aberrated = make_power_law_error(pupil_grid, 0.2, telescope_diameter)
# phase_aberrated -= np.mean(phase_aberrated)
wf_in = wf_pup.copy()
# #wf_in.electric_field *= np.exp(1j * phase_aberrated)
# #ksm.random(1e-10*k)
# #ksm.actuators[37::]=0
# ksm_num_act=int(ksm.num_actuators/3)
# random_aberations=np.random.normal(size=ksm_num_act)
# # #aber_rad = 0.5
# for i in range(ksm_num_act):
#     # ksm.set_segment_actuators(i, aber_to_opd(random_aberations[i], wavelength_wfs) / 2, 0, 0.5e-7*random_aberations[i]/ 2)
#      ksm.set_segment_actuators(i, random_aberations[i]*100E-9 / 2, 0, 0)

ksm.flatten()


# plt.figure(figsize=(8, 8))
# imshow_field(wf_fp_pistoned.phase)
# plt.colorbar()
# plt.figure()
gain = 0.4
leakage = 0.01
num_iterations = 500
deformable_mirror.flatten()
wf_fp_pistoned= ksm.forward(wf_in)
count=0
anim = FFMpegWriter('shwfs_tests.mp4', framerate=10)

total_time=0
fig=plt.figure(figsize=(8,6))

for timestep in range(num_iterations):
    layer.t = timestep*dt
    wf_in=wf.copy()

    # Propagate through atmosphere and deformable mirror.
    wf_wfs_after_atmos = ksm.forward(layer.forward((wf_in)))
    wf_wfs_after_dm = deformable_mirror.forward(wf_wfs_after_atmos)
    #just have the priamry mirror distortion
   # wf_wfs_after_dm = deformable_mirror.forward(ksm.forward(wf_in))


    # Propagate through SH-WFS
    wf_wfs_on_sh = shwfs(magnifier(wf_wfs_after_dm))
    # Read out WFS camera
    camera.integrate(wf_wfs_on_sh, dt)
    wfs_image = camera.read_out()
    #wfs_image = large_poisson(wfs_image).astype('float')

    # Calculate slopes from WFS image
    slopes = shwfse.estimate([wfs_image+1e-15])
    slopes-=np.mean(slopes)
    slopes -= slopes_ref
    slopes = slopes.ravel()

    # Perform wavefront control and set DM actuators
    deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(slopes)

    # Propagate from SM to image plane
    wf_focal = propagator.forward(wf_wfs_after_dm )


    if timestep % 25 == 0:
        plt.close(fig)
        fig=plt.figure(figsize=(8,6))

        plt.suptitle('Timestep %d / %d' % (timestep, num_iterations))

        plt.subplot(1,3,1)
        plt.title('WFS image [counts]')
        imshow_field(wfs_image, cmap='inferno')
        plt.colorbar()

        plt.subplot(1,3,2)
        plt.title('Turbulence')
        imshow_field(layer.phase_for(wavelength_wfs)*telescope_pupil, cmap='inferno') #
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.title('Instantaneous PSF [log]') #at 2.2$\\mu$m
        imshow_field(np.log10(wf_focal.intensity), cmap='inferno') #
        plt.colorbar()

        plt.tight_layout()
        anim.add_frame()
plt.close()
anim.close()
anim

