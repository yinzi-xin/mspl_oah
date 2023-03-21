import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from hcipy import *
from scipy import ndimage
from scipy import fft

def retrieve_mode(holo,ref,x1=92,y1=265,box_width=131,plot_fig=False):
    fft_holo = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(holo)))
    fft_ref = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ref)))
    
    if plot_fig:
        plt.figure()
        plt.imshow(np.abs(fft_holo),vmin=0,vmax=np.max(np.abs(fft_holo))/100)
        plt.title('FFTgram')

    fftgram_filt = np.zeros(fft_holo.shape,dtype='complex128')
    fftgram_crop = fft_holo[int(x1-box_width/2):int(x1+box_width/2),int(y1-box_width/2):int(y1+box_width/2)]
    
    if plot_fig:
        plt.figure()
        plt.imshow(np.abs(fftgram_crop))
        plt.title('Cropped FFTgram')

    xmid = int(fftgram_filt.shape[0]/2)
    ymid = int(fftgram_filt.shape[1]/2)
    fftgram_filt[int(xmid-box_width/2):int(xmid+box_width/2),int(ymid-box_width/2):int(ymid+box_width/2)] = fftgram_crop
    filter_ft = np.zeros(fft_holo.shape)
    filter_ft[xmid,ymid]=1
    fftgram_filt = ndimage.gaussian_filter(filter_ft, 19)*fftgram_filt
    
    recon_map_large = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(fftgram_filt)))/np.sqrt(ref)
    recon_map_extent = 81
    recon_pad = recon_map_large.shape[0]-recon_map_extent
    recon_map = recon_map_large[int(recon_pad/2):int(-recon_pad/2+1),int(recon_pad/2):int(-recon_pad/2+1)]
    
    cred2_pitch = 15e-6
    grid = make_uniform_grid([recon_map.shape[0],recon_map.shape[1]], [cred2_pitch*recon_map.shape[0],cred2_pitch*recon_map.shape[1]])
    recon_map_field = Field(np.transpose(recon_map).ravel(),grid)
    
    return recon_map_field
    
def retrieve_mode_lant(holo,ref,x1=92,y1=265,box_width=151,plot_fig=False):
    fft_holo = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(holo)))
    fft_ref = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ref)))
    
    if plot_fig:
        plt.figure()
        plt.imshow(np.abs(fft_holo),vmin=0,vmax=np.max(np.abs(fft_holo))/100)
        plt.title('FFTgram')

    fftgram_filt = np.zeros(fft_holo.shape,dtype='complex128')
    fftgram_crop = fft_holo[int(x1-box_width/2):int(x1+box_width/2),int(y1-box_width/2):int(y1+box_width/2)]
    
    if plot_fig:
        plt.figure()
        plt.imshow(np.abs(fftgram_crop))
        plt.title('Cropped FFTgram')

    xmid = int(fftgram_filt.shape[0]/2)
    ymid = int(fftgram_filt.shape[1]/2)
    fftgram_filt[int(xmid-box_width/2):int(xmid+box_width/2),int(ymid-box_width/2):int(ymid+box_width/2)] = fftgram_crop
    filter_ft = np.zeros(fft_holo.shape)
    filter_ft[xmid,ymid]=1
    fftgram_filt = ndimage.gaussian_filter(filter_ft, 21)*fftgram_filt
    
    recon_map_large = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(fftgram_filt)))/np.sqrt(ref)
    recon_map_extent = 101
    recon_pad = recon_map_large.shape[0]-recon_map_extent
    recon_map = recon_map_large[int(recon_pad/2):int(-recon_pad/2+1),int(recon_pad/2):int(-recon_pad/2+1)]
    
    cred2_pitch = 15e-6
    grid = make_uniform_grid([recon_map.shape[0],recon_map.shape[1]], [cred2_pitch*recon_map.shape[0],cred2_pitch*recon_map.shape[1]])
    recon_map_field = Field(np.transpose(recon_map).ravel(),grid)
    
    return recon_map_field

def find_r(recon_map_field,radii,lp_mode=0,V_num=5):
    grid = recon_map_field.grid
    coeffs = np.zeros(len(radii))
    for m in range(len(radii)):
        lp_modes = make_lp_modes(grid, V_num, radii[m])
        coeffs[m] = np.abs(np.nansum(np.conj(lp_modes[lp_mode])*recon_map_field,axis=None)/np.nansum(np.conj(lp_modes[lp_mode])*lp_modes[lp_mode],axis=None))
    best_idx = np.argmax(coeffs)
    best_r = radii[best_idx]
    return best_r    
    
def optimize_shift_location(holo,ref,x1s,y1s,box_width,lp_modes,mode_num):
    shift_coeffs = np.zeros((len(x1s),len(y1s)))
    for n in range(len(x1s)):
        for m in range(len(y1s)):
            recon_map_field = retrieve_mode(holo,ref,x1s[n],y1s[m],box_width)
            shift_coeffs[n,m] = np.abs(np.nansum(np.conj(lp_modes[mode_num])*recon_map_field,axis=None)/np.nansum(np.conj(lp_modes[mode_num])*lp_modes[mode_num],axis=None))
            
    plt.figure()
    plt.imshow(shift_coeffs)
    idxs = np.unravel_index(np.argmax(shift_coeffs),shift_coeffs.shape)
    best_x = x1s[idxs[0]]
    best_y = y1s[idxs[1]]
    
    return best_x, best_y
    
def optimize_shift_location_lant(holo,ref,x1s,y1s,box_width,lp_modes,mode_num):
    shift_coeffs = np.zeros((len(x1s),len(y1s)))
    for n in range(len(x1s)):
        for m in range(len(y1s)):
            recon_map_field = retrieve_mode_lant(holo,ref,x1s[n],y1s[m],box_width)
            shift_coeffs[n,m] = np.abs(np.nansum(np.conj(lp_modes[mode_num])*recon_map_field,axis=None)/np.nansum(np.conj(lp_modes[mode_num])*lp_modes[mode_num],axis=None))
            
    plt.figure()
    plt.imshow(shift_coeffs)
    idxs = np.unravel_index(np.argmax(shift_coeffs),shift_coeffs.shape)
    best_x = x1s[idxs[0]]
    best_y = y1s[idxs[1]]
    
    return best_x, best_y
    
def rotate_mode_to_max(ret_field,mode,plot=False):
    ret_field_2d = ret_field.shaped
    mode_2d = mode.shaped
    sq_int = np.sum(np.square(mode_2d),axis=None)
    
    grid = make_uniform_grid([mode_2d.shape[0],mode_2d.shape[1]], [mode_2d.shape[0],mode_2d.shape[1]])
    
    angles = np.linspace(0,179,180)
    
    coeffs = []
    rot_fields = []
    for angle in angles:
        rot_field_2d = ndimage.rotate(ret_field_2d,angle,reshape=False)
        sq_int = np.sum(np.square(np.abs(rot_field_2d)),axis=None)
        rot_field_2d = rot_field_2d/np.sqrt(sq_int)
        coeff = np.abs(np.nansum(np.conj(mode_2d)*rot_field_2d,axis=None)/np.nansum(np.conj(mode_2d)*mode_2d,axis=None))
        coeffs.append(coeff)
        rot_fields.append(rot_field_2d)
        
    max_idx = np.argmax(coeffs)
    rotated_field = rot_fields[max_idx]
    rot_angle = angles[max_idx]
    max_coeff = coeffs[max_idx]
    
    rot_map_field = Field(rotated_field.ravel(),grid)
    
    if plot:
        plt.figure()
        plt.plot(angles,coeffs)
        plt.xlabel('Angle')
        plt.ylabel('Coeff')
    
    return rot_map_field,rot_angle,max_coeff
    
    
def sim_mode_zernike(znums,zamps,grid,radius=14.25,best_Vnum=3.0):
    lp_modes = make_lp_modes(grid, best_Vnum, radius*15e-6)
    
    pupil_d = 0.0127*8e2
    pupil_grid = make_pupil_grid(128,diameter=pupil_d)
    prop = FraunhoferPropagator(pupil_grid, grid,focal_length=1)
    wf = Wavefront(lp_modes[0],wavelength=1550e-6)
    pup_wf = prop.backward(wf)
    zernike_modes = make_zernike_basis(11, D=pupil_d, grid=pupil_grid)
    for n in range(len(znums)):
        aber = PhaseApodizer(zernike_modes[znums[n]]*zamps[n]) #in radians
        aber_wf = aber(pup_wf)
    wf_out = prop.forward(aber_wf)
    return wf_out.electric_field
