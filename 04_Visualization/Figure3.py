#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:16:59 2020

@author: podvae01
"""
import fig_params
from fig_params import *
import HLTP_pupil
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from matplotlib import cm
from mne import viz
import mne

figures_dir = HLTP_pupil.MEG_pro_dir  +'/_figures'
freq_bands = dict(
    delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 90))  
FS_dir = HLTP_pupil.MEG_pro_dir + '/freesurfer'

                         
def plot_fig_3B():
    #Plot all the mixed model fit curves on top of binned data
    for b in ['task_prestim', 'rest']:
        res = pd.read_pickle(HLTP_pupil.result_dir + '/LM_betas_' + b + '.pkl')  
        df = pd.read_pickle(HLTP_pupil.MEG_pro_dir + 
                            '/results/roi_pwr_and_pupil' + b + '.pkl')  
        # bin data in pupil groups for presentation only
        group_percentile = np.arange(0., 100., 5);
        df['pupil_group'] =  np.digitize(df.pupil, np.percentile(df.pupil, group_percentile))
        mean_pupil_in_group = [np.mean(df.loc[df.pupil_group == g, "pupil"]) 
                for g in range(1, 21)]
        pup = np.arange(-2.5, 2.5, .1)

        
        colors = cm.winter(np.linspace(0, 255, 5).astype('int'))
        #roi_ylim = ([0.9, 1.5], [0.9, 1.5], [0.9, 1.5], [0.9, 1.5], 
         #           [0.9, 1.5], [0.9, 1.5], [0.9, 1.5])
        for roi in range(7):                

            for f, fband in enumerate(HLTP_pupil.freq_bands.keys()):
                #fig, ax = plt.subplots(figsize = [1.5, 2.]);  

                mpwr = np.zeros( (len(group_percentile) + 1, 24) )
                for sn, s in enumerate(HLTP_pupil.subjects):
                    subj_df = df.loc[df.subject == s]
                    subj_groups = np.unique(subj_df.pupil_group)
                    mpwr[subj_groups, sn] =  np.array(
                            [np.mean(subj_df.loc[subj_df.pupil_group == g, 
                                                 fband + str(roi)]) 
                        for g in subj_groups])
                m =  mpwr[1:].mean(axis = -1); 
                e = mpwr[1:].std(axis = -1)/np.sqrt(24);
                fig, ax = plt.subplots(figsize = [1.5, 2.]);  

                #plt.fill_between(mean_pupil_in_group, m -e, m + e, alpha = .2)
                plt.errorbar(mean_pupil_in_group, m, yerr = e, 
                             alpha = .3, fmt = 'o', color = colors[f])
                
                if (res.betaQpval_corrected[roi] < 0.05
                    ) | (res.betaLpval_corrected[roi] < 0.05):
                    plt.plot(pup, res[fband + 'inter'][roi] + 
                             res[fband + 'Q'][roi] * pup **2 + 
                            res[fband + 'L'][roi] * pup, 
                            linewidth = 3, zorder = 100, color = colors[f],
                            label = fband)
                plt.ylim([(m - e).min(), (m - e).min() + 0.5])
                plt.xlim([-2.5, 2.5])
                plt.locator_params(axis = 'y', nbins = 6)
                ax.spines['left'].set_position(('outward', 10))
                ax.yaxis.set_ticks_position('left')
                ax.spines['bottom'].set_position(('outward', 15))
                ax.xaxis.set_ticks_position('bottom')
                fig.savefig(figures_dir + '/roi_pwr_' + b + fband + str(roi) + 
                            '.png', bbox_inches = 'tight', transparent = True) 
                #plt.legend()
def plot_heat_maps():
    hmQ = {'task_prestim':np.zeros((7,5)), 'rest':np.zeros((7,5))}
    hmL = {'task_prestim':np.zeros((7,5)), 'rest':np.zeros((7,5))}

    for b in ['task_prestim', 'rest']:
        for f, fband in enumerate(HLTP_pupil.freq_bands.keys()):
            for roi in range(n_rois):
                mdf_Q = pd.read_pickle(HLTP_pupil.result_dir + '/LM_stats_'
                                       + str(roi) + fband + b + '.pkl')
                hmQ[b][roi, f] = mdf_Q.params[1]
                hmL[b][roi, f] = mdf_Q.params[2]
        fig, ax = plt.subplots(1, 1,figsize = (1.5,3))     
        
        plt.imshow(hmQ[b], cmap = 'Spectral_r', vmin = -0.03, vmax = 0.03)
        ax.axis('off')
        fig.savefig(figures_dir + '/roi_betasQ_' + b + '.png', 
                bbox_inches = 'tight', transparent = True)
        fig, ax = plt.subplots(1, 1,figsize = (1.5,3))     

        plt.imshow(hmL[b], cmap = 'Spectral_r', vmin = -0.05, vmax = 0.05)
        ax.axis('off')
        fig.savefig(figures_dir + '/roi_betasL_' + b + '.png', 
                bbox_inches = 'tight', transparent = True)
    # compare
    fig, ax = plt.subplots(1, 1,figsize = (1.5,3))     
    plt.imshow(hmQ['task_prestim'] - hmQ['rest'], 
               cmap = 'Spectral_r', vmin = -0.03, vmax = 0.03)
    ax.axis('off')
    fig.savefig(figures_dir + '/roi_betasQ_diff.png', 
                bbox_inches = 'tight', transparent = True)
    
    fig, ax = plt.subplots(1, 1,figsize = (1.5,3))     
    plt.imshow(hmL['task_prestim'] - hmL['rest'], 
               cmap = 'Spectral_r', vmin = -0.05, vmax = 0.03)
    ax.axis('off')
    fig.savefig(figures_dir + '/roi_betasL_diff.png', 
                bbox_inches = 'tight', transparent = True)
                
def plot_fig_3C():
    colors = cm.winter(np.linspace(0, 255, 5).astype('int'))

    for b in ['task_prestim', 'rest']:
        res = pd.read_pickle(HLTP_pupil.result_dir + '/LM_betas_' + \
                             b + '.pkl')            
        
        for roi in range(7):
            fig, ax = plt.subplots(1, 1,figsize = (1.5,2))     
            k = 0
            for term in ['Q', 'L']:
                for ii, fband in enumerate(HLTP_pupil.freq_bands.keys()):
                     
                    plt.bar(k, res[fband + term][roi], 
                            yerr = res[fband + term + 'err'][roi], 
                            facecolor = colors[ii], edgecolor = 'none', 
                            ecolor = colors[ii],
                            capsize = 2)
                    k += 1
            plt.ylim([-.06, .06])
            plt.locator_params(axis = 'y', nbins = 6)

            ax.spines["top"].set_visible(False); 
            ax.spines["right"].set_visible(False) 
            ax.spines["bottom"].set_visible(False) 
            ax.spines['left'].set_position(('outward', 10))
            ax.yaxis.set_ticks_position('left')
            fig.savefig(figures_dir + '/roi_betas_' + b + str(roi) + 
                            '.png', bbox_inches = 'tight', transparent = True)
            #ax.spines['bottom'].set_position(('outward', 15))
            #ax.xaxis.set_ticks_position('bottom')
            #plt.ylabel('', fontsize = 14)
            #plt.xlabel('Freq. band', fontsize = 14)
from surfer import Brain, project_volume_data
            
def plot_RSN_atlas():
    yeo = datasets.fetch_atlas_yeo_2011()
    atlas_yeo = yeo.thick_7

    fsaverage = datasets.fetch_surf_fsaverage()
    imgfig = plotting.plot_roi(atlas_yeo, 
                  cut_coords=(8, -4, 9), colorbar=True, cmap='Paired')
    imgfig.savefig(figures_dir + '/atlas_roi.png')
    
    atlas_name = 'Yeo2011_7Networks_N1000'
    brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=FS_dir,
               cortex='low_contrast', background='white', size=(800, 600))
    brain.add_annotation(atlas_name)
plot_fig_3A('task_prestim_ds')
plot_fig_3A('mean_rest')
plot_fig_3B()





## Perhaps add in the future:

results_dir = HLTP_pupil.MEG_pro_dir + '/results'
epoch_name = 'task_prestim_ds'
band = 'beta'
band_stc = mne.read_source_estimate(results_dir + '/' +  epoch_name + 
                                    '_src_1wANOVA_'  + band) 
fig = viz.plot_source_estimates(band_stc, subjects_dir=FS_dir, 
                    subject='fsaverage', colorbar = False, surface = 'inflated',
                    hemi='rh', transparent = False, background = 'white',
                    time_label='', views='lat', alpha = 0.9,
                    colormap = 'jet', backend = 'matplotlib')
                                    