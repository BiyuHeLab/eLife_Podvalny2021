#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:55:49 2020

@author: podvae01
"""
#import sys
#sys.path.append('../../')
import HLTP_pupil
#from mayavi import mlab

from HLTP_pupil import MEG_pro_dir
import numpy as np
from mne import viz
figures_dir = HLTP_pupil.MEG_pro_dir  +'/_figures'

ename = 'mean_rest'#'rest02_ds'#'task_prestim_ds'##'rest02_ds'#  
norm_stcs = HLTP_pupil.load(MEG_pro_dir + '/pupil_result/norm_stc_' + ename)
raw_stcs = HLTP_pupil.load(MEG_pro_dir + '/pupil_result/raw_stc_' + ename)
freq_bands = dict(
    delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 90))  
FS_dir = HLTP_pupil.MEG_pro_dir + '/freesurfer'

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
fband  =  4; band = 'gamma'
grp = 1
band_stc = stc_sub_mean[grp].copy().crop(fband, fband).mean() 

## TEST 1
#mlab.options.offscreen = True
#fig = mlab.figure(size=(300, 300))
#fig.scene.off_screen_rendering = True
#band_stc.plot(subjects_dir=FS_dir, title = band + str(grp),
#                        subject='fsaverage', figure = fig, colorbar = False,
#                        hemi='both', transparent = False, background = 'white',
#                        time_label='', views='lateral', alpha = 0.9,
#                        colormap = 'RdYlBu_r', 
#                clim=dict(kind='value', lims=(-.1, 0., .1)))
#fig.scene.off_screen_rendering = True
#
#fig.scene.save_png(figures_dir + '/test1' + ename + '_raw_fbands_src_' 
#                     + band + str(grp) + '.png')

## TEST 2    

#mlab.options.offscreen = True
fig = viz.plot_source_estimates(band_stc, subjects_dir=FS_dir, subject='fsaverage',
                                colorbar = False,
                        hemi='lh', transparent = False, background = 'white',
                        time_label='', views='lat', alpha = 0.9,
                       colormap = 'RdYlBu_r', 
                clim=dict(kind='value', lims=(-.1, 0., .1)))



fig.savefig(figures_dir + '/test1' + ename + '_raw_fbands_src_' 
                     + band + str(grp) + '.png')                              
                              
                              
                              
                              
                              
    