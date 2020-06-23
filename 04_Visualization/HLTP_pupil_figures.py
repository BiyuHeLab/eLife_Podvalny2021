#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:13:50 2019

@author: podvae01
"""
import sys
sys.path.append('../../')
from os import path
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir
from mayavi import mlab
from scipy import stats
import numpy as np
from matplotlib import rcParams, cm
import matplotlib.pyplot as plt
import mne
from mne import viz
import numpy as np
import pandas as pd
import pickle
mlab.options.offscreen = True

results_dir = HLTP_pupil.MEG_pro_dir + '/results'
figures_dir = HLTP_pupil.MEG_pro_dir  +'/_figures'

colors = {'Rec':np.array([255, 221, 140]) / 255., 
          'Unr':np.array([164, 210, 255]) / 255., 
          'All':np.array([174, 236, 131]) / 255.,
          'RecD':np.array([255, 147, 0]) / 255., 
          'UnrD':np.array([4, 51, 255]) / 255., 
          'AllL':np.array([195, 250, 160]) / 255.,
          'AllD':np.array([127, 202, 96]) / 255.,
          'AllDD':np.array([53, 120, 33]) / 255.,
          'sens':np.array([244, 170, 59]) / 255.,
          'bias':np.array([130, 65, 252]) / 255.    }

fig_width = 7  # width in inches
fig_height = 4.2  # height in inches
fig_size =  [fig_width,fig_height]
params = {    
          'axes.spines.right': False,
          'axes.spines.top': False,
          
          'figure.figsize': fig_size,
          
          'ytick.major.size' : 8,      # major tick size in points
          'xtick.major.size' : 8,    # major tick size in points
              
          'lines.markersize': 6.0,
          # font size
          'axes.labelsize': 14,
          'axes.titlesize': 14,     
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'font.size': 12,

          # linewidth
          'axes.linewidth' : 1.3,
          'patch.linewidth': 1.3,
          
          'ytick.major.width': 1.3,
          'xtick.major.width': 1.3,
          'savefig.dpi' : 800
          }
rcParams.update(params)
rcParams['font.sans-serif'] = 'Helvetica'

freq_bands = dict(
    delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 90))  
FS_dir = HLTP_pupil.MEG_pro_dir + '/freesurfer'
group_percentile = np.arange(0., 100., 20);

# -----------------------------------------------------------------------------
# -----PLOT EXAMPLE PUPIL RECORDINS -------------------------------------------
# -----FIGURE 1----------------------------------------------------------------
# -----------------------------------------------------------------------------

subject = 'DJ'; block_name = 'rest02'
raw_pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                        '/raw_pupil_' + block_name + '.pkl')
pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                        '/clean_pupil_' + block_name + '.pkl')
int_pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                        '/clean_interp_pupil_' + block_name + '.pkl')

fs = 1200
x = 80 *  fs; nsec = 60
fig, ax = plt.subplots(1, 1, figsize = (2.5,1.5))
plt.plot(np.linspace(0, nsec, fs * nsec), raw_pupil_data[x:(x+ fs * nsec)], 
         color= [.8,.8,.8]);
plt.plot(np.linspace(0, nsec, fs * nsec), int_pupil_data[x:(x+ fs * nsec)])
plt.plot(np.linspace(0, nsec, fs * nsec), pupil_data[x:(x+ fs * nsec)])
plt.ylim([-4, -2]);plt.xlim([0, nsec]); 
plt.xlabel('Time (s)');plt.ylabel('Pupil size (a.u.)')
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
fig.savefig(figures_dir + '/example_pupil_' + subject + '.png', 
                bbox_inches = 'tight', transparent=True)

# -----PLOT EXAMPLE PUPIL and MEG RECORDINS -----------------------

fname = HLTP_pupil.MEG_pro_dir + '/' + subject + \
                              '/' + block_name + '_stage2_raw.fif'
from scipy.signal import butter, lfilter
from scipy import signal
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a                              
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y    
raw = mne.io.read_raw_fif(fname, preload = True)
picks = mne.pick_types(raw.info, meg = True, 
                           ref_meg = False, exclude = 'bads')
raw.apply_function(signal.detrend, picks=picks, dtype=None, n_jobs=24)
data = raw.get_data()

meg_data =  butter_lowpass_filter(data[249, :], 5, HLTP_pupil.raw_fs)
filt_pupil = butter_lowpass_filter(int_pupil_data, 5,HLTP_pupil.raw_fs )
fs = 1200
x = 80 *  fs; nsec = 60
fig, ax = plt.subplots(1, 1, figsize = (3.,1.5))

plt.plot(np.linspace(0, nsec, fs * nsec), zscore(meg_data[x:(x+ fs * nsec)]))
plt.plot(np.linspace(0, nsec, fs * nsec), zscore(filt_pupil[x:(x+ fs * nsec)]))
print(scipy.stats.pearsonr( zscore(meg_data[x:(x+ fs * nsec)]),
                            zscore(filt_pupil[x:(x+ fs * nsec)] )))
plt.ylim([-5, 5]);plt.xlim([0, nsec]); 
plt.xlabel('Time (s)');plt.ylabel('zscore')
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')


              
# plot power maps
ename = 'mean_rest'#'rest02_ds'#'task_prestim_ds'##'rest02_ds'#  
norm_stcs = HLTP_pupil.load(MEG_pro_dir + '/pupil_result/norm_stc_' + ename)
raw_stcs = HLTP_pupil.load(MEG_pro_dir + '/pupil_result/raw_stc_' + ename)

stcs = norm_stcs
n_subjects = len(norm_stcs)
stc_sub_mean = {}
for grp in range(1,6):
    data = np.array([stcs[s][str(grp)].data for s in range(0, n_subjects)])
    #tstat = stats.ttest_1samp(data, popmean = 0, axis = 0)[0]        
    stc_sub_mean[grp] = stcs[0][str(grp)].copy()
    stc_sub_mean[grp]._data = data.mean(axis = 0)#tstat

# Plot mean power   
# plot for each frequency band
for fband, band in enumerate(freq_bands.keys()):
    #if fband >1: continue;
    for grp in range(1, 6):
        #if grp >1: continue;
        band_stc = stc_sub_mean[grp].copy().crop(fband, fband).mean() 
        fig = mlab.figure(size=(300, 300))
        band_stc.plot(subjects_dir=FS_dir, title = band + str(grp),
                        subject='fsaverage', figure = fig, colorbar = False,
                        hemi='both', transparent = False, background = 'white',
                        time_label='', views='lateral', alpha = 0.9,
                        colormap = 'RdYlBu_r', 
                clim=dict(kind='value', lims=(-.1, 0., .1)))
        #fig.scene.off_screen_rendering = True

        fig.scene.save_png(figures_dir + '/MEG_' + ename + '_raw_fbands_src_' 
                     + band + str(grp) + '.png')
# plot power in ROIs
labels = {}
for hs in ['lh', 'rh']:
    labels[hs] = mne.read_labels_from_annot(
        'fsaverage', 'Yeo2011_7Networks_N1000', hs, subjects_dir=FS_dir)

rois = [label.name for label in labels[hs]];rois = rois[:-1]
roi_idx = []
for i, r in enumerate(rois):
    roi_idx.append(np.where(np.array([ l.name == rois[i] 
        for l in labels[hs] ]))[0][0])
    
for ename in ['mean_rest', 'task_prestim_ds']: #'rest01_ds', 'rest02_ds',
           
    data = HLTP_pupil.load(MEG_pro_dir + 
                     '/pupil_result/roi_power_map_7nets_' + ename)            
    n_subjects = data.shape[-1]
    n_roi = len(roi_idx)
    for fband, band in enumerate(freq_bands.keys()):
        c=cm.brg(np.linspace(0,1,n_roi))
        fig, ax = plt.subplots(figsize = [2., 2.]);      
        for i in roi_idx:
            mdata = np.expand_dims(data[fband,:, i, :].mean(axis = -1), 
                                   axis = 1)
            edata = np.expand_dims(data[fband,:, i, :].std(axis = -1), 
                                   axis = 1) / np.sqrt(n_subjects)
            plt.errorbar(range(5), mdata, yerr = edata, capsize = 3.,linewidth = 2.,
                         label = i, color = c[i, :], alpha = .6);#labels['rh'][i].color
        plt.ylabel('relative power (dB)')
        plt.xlabel('Pupil size')
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.ylim([-0.1, 0.1]);plt.xlim([0, 4])
        plt.plot([0,4], [0, 0], '--k')
        plt.title(band)
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=7)
        #plt.legend(bbox_to_anchor=(1.5, 1.05))
        #ax = plt.gca()
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        #ax.set_facecolor('gray')
        fig.savefig(figures_dir + '/MEG_' + ename + band + '_7by_pupil_size.png', 
                               bbox_inches = 'tight',  dpi = 800, transparent = True)
           
    
# plot matrix powermaps in ROI: 
epoch_name = 'task_prestim_ds'
      
power_map = HLTP_pupil.load(MEG_pro_dir + 
                    '/pupil_result/group_roi_power_map_7nets_' + epoch_name)
from numpy.ma import masked_array
n_roi = 7
for roi in range(n_roi):
    fig, ax = plt.subplots(figsize = [2., 3.]);  
    tstt, pval=stats.ttest_1samp(power_map[roi, :, :, :], 
                                 popmean = 0, axis= -1 )    
    sig = masked_array(tstt, pval<0.05)
    nsig = masked_array(tstt, pval>=0.05)
    plt.imshow(tstt.T, interpolation = 'hamming',
                   vmin = -4., vmax = 4., cmap = 'inferno', 
                  extent = [0, 100, 100, 1]);     
    ax.grid(b=True, which='minor', color='w', linestyle='--')
    ax.grid(b=True, which='major', color='w', linestyle='--')

    ax.axis('on')
    plt.box(True)
    ax.spines['top'].set_visible(True);    ax.spines['right'].set_visible(True)

               #plt.imshow(sig.T, interpolation = 'hamming',
    #               vmin = -3., vmax = 3., cmap = 'RdYlBu_r', 
    #              extent = [-10, 10, 100, 1]);
    plt.yscale('log'); #plt.xscale('log'); 
    plt.xlabel('Pupil size (a.u.)')
    plt.ylabel('Frequency (Hz)')
    ax.locator_params(axis='x', nbins=5)
    fig.savefig(figures_dir + '/power_map_7by_pupil_size' 
                + str(roi) + '.png', bbox_inches = 'tight', 
                dpi = 800, transparent = True)    

plt.colorbar()      
fig.savefig(figures_dir + '/power_map_7by_pupil_size_cmap_.png', 
            bbox_inches = 'tight', 
                dpi = 800, transparent = True) 
from scipy.optimize import curve_fit
def ushape(x, a,b,c):
    return a*x**2 + b*x + c
freq_bands = dict(delta=(1, 5), alpha=(7, 15), gamma=(20, 100))  
color=cm.plasma(np.linspace(0, 1, 4))
for roi in range(n_roi):
    fig, ax = plt.subplots(figsize = [2., 3.]);  
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    for fband, band in enumerate(freq_bands.keys()):
        freqs = [(freq_bands[band][0] - 1), (freq_bands[band][1] )]
        pwr = power_map[roi, :, freqs[0]:freqs[1], :].mean(axis = 1)
        mean_pwr = pwr.mean(axis = -1)
        error_pwr = stats.sem(pwr, axis = -1)
        pupil = np.linspace(0, 100, len(mean_pwr))
        plt.fill_between(pupil, 
                         mean_pwr- error_pwr, mean_pwr+ error_pwr, 
                         alpha =.3,color = color[fband] )

        popt, pcov = curve_fit(ushape, np.tile(np.expand_dims(pupil, axis = 0).T, 24).flatten(), pwr.flatten())
        plt.plot(pupil, ushape(pupil, popt[0], popt[1], popt[2]), color = color[fband], linewidth = 3)
    plt.ylabel('relative power (dB)')
    plt.xlabel('pupil size (%)')    
    plt.xlim([0, 100]); plt.ylim([-.15,.15]);
    fig.savefig(figures_dir + '/power_bands_7by_pupil_size' 
                + str(roi) + '.png', bbox_inches = 'tight', 
                dpi = 800, transparent = True)        
    
# -----PLOT DISTRIBUTION OF PUPIL SIZES COLORED BY GROUP-----------------------
pupil_df = pd.read_pickle(HLTP_pupil.result_dir +'/pupil_rest_df.pkl')
pupil_df = pupil_df.dropna()
all_spupil = []
for subject in HLTP_pupil.subjects:
    spupil = stats.zscore(pupil_df[pupil_df.subject == subject].pupil.values)
    all_spupil.append(spupil)
    sgroup = pupil_df[pupil_df.subject == subject].group.values
    perc = np.percentile(spupil, group_percentile)
    group = np.digitize(spupil, perc, right = True)
    
    fig, ax = plt.subplots(1, 1, figsize = (3.5,2.5))
    
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')

    plt.hist(spupil, bins = np.arange(-3, 3.01, .5), color = 'gray')
    for p in perc[1:-1]:
        plt.axvline(p, color='k', linestyle=':')
    plt.ylim([0, 40]); plt.xlim([-4, 4])
    plt.locator_params(axis='x', nbins=7)

    plt.ylabel('# of samples')
    plt.xlabel('pupil size (zscore)')
    fig.savefig(figures_dir + '/example_hist_pupil_rest' + subject + '.png', 
                bbox_inches = 'tight', transparent=True)

fig, ax = plt.subplots(1, 1, figsize = (3.5,2.5))
    
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
plt.hist(np.concatenate(all_spupil), bins = np.arange(-4, 4.01, .25), color = 'gray')
plt.ylim([0, 300]); plt.xlim([-4, 4])
plt.locator_params(axis='x', nbins=7)
plt.ylabel('# of samples')
plt.xlabel('pupil size (zscore)')
fig.savefig(figures_dir + '/example_hist_pupil_rest_all_subj.png', 
            bbox_inches = 'tight', transparent=True)

# ---------------------------------------------------------------------
for grp in range(1,6):
    for fband, band in enumerate(freq_bands.keys()): 
        
        band_stc= stc_sub_mean[grp].copy().crop(fband, fband).mean() 

        fig = mlab.figure(size=(300, 300))
        #fig.scene.off_screen_rendering = True 
        band_stc.plot(subjects_dir=FS_dir, title = band + str(grp),
                        subject='fsaverage', figure = fig, background = 'white',
                        hemi='both', transparent = False, colormap ='mne',
                        time_label='', views='ventral', alpha = 0.9,
                        clim=dict(kind='value', lims=[-.4, 0, 0.4]))
        fig.scene.off_screen_rendering = True
        mlab.savefig(figures_dir + 
                     '/MEG_rest_pupil_fband_bhv_ventral' + band+'.png')