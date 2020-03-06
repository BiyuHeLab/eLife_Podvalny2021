#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:09:21 2019

@author: podvae01
"""
import conpy, mne  # Import required Python modules
import HLTP_pupil
from HLTP_pupil import subjects, freq_bands, MEG_pro_dir
import numpy as np
import pandas as pd
method = 'dics'
pupil_groups = np.arange(1, 6)
FS_dir= MEG_pro_dir  + '/freesurfer'
figures_dir = MEG_pro_dir  + '/_figures'
results_dir = MEG_pro_dir  +'/results'

epoch_name = 'task_prestim_ds'
# calculate spectral connectivity for each subject 
for subject in subjects:
    sub_pro_dir = MEG_pro_dir + '/' + subject
    fwd = mne.read_forward_solution(sub_pro_dir + '/HLTP_fwd.fif')
    fwd_tan = conpy.forward_to_tangential(fwd)
    pairs = conpy.all_to_all_connectivity_pairs(fwd_tan, min_dist = 0.04)
    for pupil_group in pupil_groups:
        fname = sub_pro_dir + '/' + method + \
            '_csd_multitaper_' + epoch_name + str(pupil_group)
        #fname = sub_pro_dir + '/' + method + '_csd_pupil' + str(pupil_group)
        csd = mne.time_frequency.read_csd(fname)
        for band in freq_bands.keys():
            csd_band = csd.mean(fmin = freq_bands[band][0],
                                fmax = freq_bands[band][1]) 
            # this step takes lots of time
            con = conpy.dics_connectivity(pairs, fwd_tan, csd_band, 
                  n_jobs = -1, reg = 0.05, n_angles = 50)
            con.save(sub_pro_dir + '/' + method + '_con_pupil_' + epoch_name +
                     str(pupil_group) + '_fband_' + band)
            
# translate to fsaverage - DOESNT WORK NOW
fdir = '/isilon/LFMI/VMdrive/Ella/mne_data/MNE-sample-data/subjects/'            
src_fname = fdir + 'fsaverage/bem/fsaverage-ico-5-src.fif'
fsaverage = mne.read_source_spaces(src_fname)
for subject in subjects:
    sub_pro_dir = HLTP_pupil.MEG_pro_dir + '/' + subject
    for pupil_group in pupil_groups:
        for band in freq_bands.keys():
            con = conpy.read_connectivity(sub_pro_dir + '/' + method + 
                        '_con_pupil_' + epoch_name +
                             str(pupil_group) + '_fband_' + band)     
            con.to_original_src(fsaverage)
            
# parcelate the connectivity for each subject according to FS labels      
p_all = []                        
for subject in subjects:
    sub_pro_dir = HLTP_pupil.MEG_pro_dir + '/' + subject
    labels = mne.read_labels_from_annot(subject, 'aparc', subjects_dir=FS_dir)
    for pupil_group in pupil_groups:
        for band in freq_bands.keys():
            fname = sub_pro_dir + '/' + method + '_con_pupil_' + epoch_name + \
                str(pupil_group) + '_fband_' + band
            con = conpy.read_connectivity( fname)
            con.subject = labels[0].subject
            # in this parcellation for each voxel, we select the strongest 
            # connection and then average across connections within parcel.  
            p = con.parcellate(labels, summary = 'absmax', 
                               weight_by_degree = False)
            
            pd_df = pd.DataFrame({ 'conn':[], 'pair1':[],'pair2':[], 'subject':[], 
                      'pupil_group':[], 'fband':[]}  )
            pd_df['conn']=p.data; 
            pd_df['subject'] = subject;
            pd_df['fband'] = band;
            pd_df['pupil_group'] = pupil_group; 
            pd_df['pair1']=p.pairs[0]; 
            pd_df['pair2']=p.pairs[1]
            #pd_df['pairlabel1']=labels(p.pairs[0]); pd_df['pairlabel2']=p.pairs[1]
            p_all.append(pd_df.copy())
df = pd.concat(p_all)
df.to_pickle( results_dir + '/group_conn_df' )
from matplotlib import cm
# average across subjects
n_con = len(labels)
for band in list(freq_bands.keys())[1:]:
    mats = []        
    p_mat = np.zeros([68, 68]);    F_mat = np.zeros([68, 68])
    p_mat[:] = np.nan; F_mat[:] = np.nan; 
    pairs = [];mean_con = [];p.data = [];p.pairs = [];
    for p1 in range(n_con):
        for p2 in np.arange(p1 + 1, n_con):
            data = []
            for pupil_group in pupil_groups:
                data.append(df[(df.fband == band) & 
                               (df.pupil_group == pupil_group) & 
                   (df.pair1 == p1) & (df.pair2 == p2)].conn.values)
            F_mat[p1, p2], p_mat[p1, p2] = np.array(
                    mne.stats.f_mway_rm(np.array(data).T, [5], 
                                        effects = 'A', return_pvals = True))
    sig_pairs = np.where( p_mat < 0.001  )   
            
    p.data = F_mat[sig_pairs];
    p.pairs = sig_pairs;
    
    fig = plt.figure()
    p.plot(n_lines = 25, vmin = 0., vmax = 10., facecolor = 'white',
           node_edgecolor = 'none',textcolor = 'black', 
           node_colors = cm.jet(np.linspace(0, 255, n_con).astype('int')),
                       colormap = 'gray_r', fig = fig)
    fig.savefig(figures_dir + '/conn_pupil_fbands_' + band + '.png', 
                            dpi = 800,  bbox_inches = 'tight',transparent = True)
    

mc = get_label_conn_as_mat(p)
lnames = [label.name for label in labels];
values = range(68)
fig = plt.figure(figsize = (10,10))
plt.imshow(mc, cmap = 'gray')
plt.xticks(values, lnames, fontsize=14, **HLTP.hfont)
plt.yticks(values, lnames, fontsize=14, **HLTP.hfont) 

def get_label_conn_as_mat(p):
    mat_con = np.zeros((p.n_sources, p.n_sources))
    for conn in range(p.n_connections):
        mat_con[p.pairs[0, conn],p.pairs[1, conn]] = p.data[conn]
    return mat_con
# visualize
for subject in subjects:
    sub_pro_dir = HLTP.MEG_pro_dir + '/' + subject
    for pupil_group in pupil_groups:
        for band in freq_bands.keys():     
            p.plot(n_lines=20, vmin=0.8, vmax=1, facecolor='white', 
                   textcolor='black', colormap='gray_r')
    






                  
# TESTS transfor the connectivity to fsaverage??
src_fname = FS_dir + '/fsaverage-ico6-src.fif'
fsaverage = mne.read_source_spaces(src_fname)    
os.environ["SUBJECTS_DIR"] = FS_dir
src_avg = mne.setup_source_space('fsaverage', spacing = 'ico6')     
con_test = con.to_original_src(src_orig = src_avg, subjects_dir=FS_dir)

labels = mne.read_labels_from_annot('AW', 'aparc', subjects_dir=FS_dir)
p=con.parcellate(labels, 'absmax', weight_by_degree=False)
p.plot(n_lines=20, vmin=0, vmax=1)

stc_fsaverage = stc.morph('fsaverage', subjects_dir=HLTP.MRI_dir)
stc_fsaverage.save(sub_pro_dir + '/fsaverage_' + fname)

            


