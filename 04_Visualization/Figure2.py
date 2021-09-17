#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:57:49 2020

@author: podvae01
"""
import fig_params
from fig_params import *
import HLTP_pupil
import numpy as np
from mne import viz
from mayavi import mlab
# use either mne viz or mayavi, whatever works - can be buggy with Virtual Machines
figures_dir = HLTP_pupil.MEG_pro_dir  +'/_figures'
freq_bands = dict(
    delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 90))  
FS_dir = HLTP_pupil.MEG_pro_dir + '/freesurfer'

def plot_fig_2A(ename):
    # plot DICS power on brain surface
    norm_stcs = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + 
                            '/pupil_result/norm_stc_' + ename)
    
    stcs = norm_stcs
    n_subjects = len(norm_stcs)
    stc_sub_mean = {}
    for grp in range(1,6):
        data = np.array([stcs[s][str(grp)].data for s in range(0, n_subjects)])
        stc_sub_mean[grp] = stcs[0][str(grp)].copy()
        stc_sub_mean[grp]._data = data.mean(axis = 0)
    todb = 10/np.log(10)
    # Plot mean power for each frequency band
    for fband, band in enumerate(freq_bands.keys()):
        for grp in range(1, 6):
            band_stc = todb * stc_sub_mean[grp].copy().crop(fband, fband).mean() 
            # fig = viz.plot_source_estimates(band_stc, subjects_dir=FS_dir, 
            #         subject='fsaverage', colorbar = False, surface = 'inflated',
            #         hemi='rh', transparent = False, background = 'white',
            #         time_label='', views='lat', alpha = 0.9,
            #         colormap = 'RdYlBu_r', backend = 'matplotlib',
            #         clim=dict(kind = 'value', lims = (-.15, 0., .15)))
    
            # fig.savefig(figures_dir + '/DICS' + ename + '_raw_fbands_src_' 
            #              + band + str(grp) + '.png', 
            #         bbox_inches = 'tight', transparent=True)
            
            fig = mlab.figure(size=(300, 300))
            tsts = band_stc.plot(subjects_dir=FS_dir, title = band + str(grp), 
                          background = 'white', alpha = 0.9,
                        subject='fsaverage', figure = fig, colormap = 'RdYlBu_r',
                        hemi='both', transparent = False, backend = 'mayavi',
                        time_label='', views='lateral', 
                clim=dict(kind='value', lims=(-0.5, 0., 0.5)))
            
            fig.off_screen_rendering = True
            #fig.save_image(figures_dir + '/testMEG_pupil_fbands_src_' + band + str(grp) + '.png')

            mlab.savefig(figures_dir + '/DICS' + ename + '_raw_fbands_src_' 
                         + band + str(grp) + '.png')
            mlab.close(all = True)
            
plot_fig_2A('task_prestim_ds')
plot_fig_2A('mean_rest')            