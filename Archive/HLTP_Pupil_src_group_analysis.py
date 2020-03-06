#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:38:22 2019

@author: podvae01
"""
import HLTP_pupil
import numpy as np
import pandas as pd
import mne
from mayavi import mlab

figures_dir = HLTP_pupil.MEG_pro_dir  +'/_figures'
results_dir = HLTP_pupil.MEG_pro_dir  +'/results'
df = pd.read_pickle(HLTP_pupil.MEG_pro_dir + 
                               '/results/SDT_bhv_pupil_df.pkl')
FS_dir = HLTP_pupil.MEG_pro_dir + '/freesurfer'
#freq_bands = dict(
#    delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 90))    

freq_bands = dict(gamma=(30, 90)) 

subjects = ['AA', 'AC', 'AL', 'AR', 'BJB', 'CW', 'DJ', 'EC', 'FSM','JA',
                    'JP', 'JS', 'LS', 'NA', 'NM', 'MC', 'SM', 'TL', 'TK']
method = 'dics'
tag = 'rest_pupil'#'pupil'
# normalize each subject by average across groups 
stcs = {'1':[],'2':[],'3':[],'4':[],'5':[]}
for s, subject in enumerate(subjects): 
    sub_pro_dir = HLTP_pupil.MEG_pro_dir + '/' + subject
    for grp in range(1,6):
        fname = method + '_power_map_' + tag + str(grp)
        stc_fname = sub_pro_dir + '/fsaverage_' + fname
        stcs[str(grp)].append(mne.read_source_estimate(stc_fname))
    stc_group_mean = stcs['1'][s].copy()
    for grp in range(2,6):
        stc_group_mean += stcs[str(grp)][s].data
    stc_group_mean /= 5
    for grp in range(1,6):
        stcs[str(grp)][s] /= stc_group_mean
        stcs[str(grp)][s]._data = np.log(stcs[str(grp)][s]._data)
        
# calculate mean normalized power across subjects for each group              
n_subjects = len(stcs[str(grp)])
stc_sub_mean = {}
for grp in range(1,6):
    stc_sub_mean[grp] = stcs[str(grp)][0].copy()
    for s in range(1, n_subjects):#1 is here because of the line above - looks bad
        stc_sub_mean[grp]._data += stcs[str(grp)][s].data
    stc_sub_mean[grp]._data /= n_subjects 

# calculate correlation between power and bhv by pupil size in each voxel
#n_voxels = 20484
#for bhv in ['HR', 'FAR','c','d']:
#    for fband, band in enumerate(freq_bands.keys()):
#        #subjects X pupil_group X voxels 
#        X = []
#        for grp in range(1,6):
#            X_grp = np.zeros((n_subjects, 20484))
#            for s in range(n_subjects):
#                X_grp[s, :] = stcs[str(grp)][s]._data[:, fband]
#                
#            X.append(X_grp)    
#        X = np.array(X)
#        rho = np.zeros((n_subjects, n_voxels))
#        for s_id, s in enumerate(subjects):
#            print(s, bhv, band)
#            for v in range(20484):
#                rho[s_id, v], _ = scipy.stats.spearmanr(
#                        df[df.subject == s][bhv].values, X[:, s_id, v])
#        HLTP.save(rho, results_dir + '/rho_MEG_pupil_fband_bhv_' + band+ bhv)
        
for bhv in ['HR', 'FAR','c','d']:
    for fband, band in enumerate(freq_bands.keys()): 
        rho = HLTP.load(results_dir + '/rho_MEG_pupil_fband_bhv_' + band+ bhv)
        band_stc= stc_sub_mean[grp].copy().crop(fband, fband).mean() 
        band_stc._data = rho.mean(axis = 0)

        fig = mlab.figure(size=(300, 300))
        #fig.scene.off_screen_rendering = True 
        band_stc.plot(subjects_dir=FS_dir, title = band + str(grp),
                        subject='fsaverage', figure = fig, background = 'white',
                        hemi='both', transparent = False, colormap ='mne',
                        time_label='', views='ventral', alpha = 0.9,
                        clim=dict(kind='value', lims=[-.4, 0, 0.4]))
        fig.scene.off_screen_rendering = True
        mlab.savefig(figures_dir + 
                     '/MEG_pupil_fband_bhv_ventral' + band+ bhv+'.png')
    
    
    