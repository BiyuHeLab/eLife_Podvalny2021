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
from scipy.stats import zscore
#from mne import viz
#import mne
from nilearn import datasets
from nilearn import plotting, surface
#from surfer import Brain, project_volume_data

figures_dir = HLTP_pupil.MEG_pro_dir  +'/_figures'
freq_bands = dict(
    delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 90))  
FS_dir = HLTP_pupil.MEG_pro_dir + '/freesurfer'

def plot_fig_3A():

    fsaverage = datasets.fetch_surf_fsaverage()
    hemi = 'right'
    fname = "/home/podvae01/Downloads/Yeo_JNeurophysiol11_FreeSurfer/fsaverage5/label/rh.Yeo2011_7Networks_N1000.annot"
    views = ['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior']

    for v in views:
        fig = plotting.plot_surf_roi(fsaverage['pial_' + hemi], roi_map=fname,
                           hemi=hemi, view=v, cmap = 'Pastel1', darkness= 0.5,
                           bg_map=fsaverage['sulc_' + hemi], bg_on_data=True)
        fig.savefig(figures_dir +  '/yeo7atlas_' + hemi + v + '.png', dpi = 600)


    hemi = 'left'
    fname = "/home/podvae01/Downloads/Yeo_JNeurophysiol11_FreeSurfer/fsaverage5/label/lh.Yeo2011_7Networks_N1000.annot"
    for v in views:
        fig = plotting.plot_surf_roi(fsaverage['pial_' + hemi], roi_map=fname,
                           hemi=hemi, view=v, cmap = 'Pastel1', darkness = 0.5,
                           bg_map=fsaverage['sulc_' + hemi], bg_on_data=True)
        fig.savefig(figures_dir +  '/yeo7atlas_' + hemi + v + '.png', dpi = 600)

def plot_RSN_atlas():
    yeo = datasets.fetch_atlas_yeo_2011(
        data_dir = '/isilon/LFMI/VMdrive/Ella/nilearn_data/')
    atlas_yeo = yeo.thick_7

    #fsaverage = datasets.fetch_surf_fsaverage()
    imgfig = plotting.plot_roi(atlas_yeo, 
                  cut_coords=(8, -4, 9), colorbar=True, cmap='Paired')
    imgfig.savefig(figures_dir + '/atlas_roi.png')
    
    atlas_name = 'Yeo2011_7Networks_N1000'
    brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=FS_dir,
               cortex='low_contrast', background='white', size=(800, 600))
    brain.add_annotation(atlas_name)
        
def plot_fig_3B():
    #Plot all the mixed model fit curves on top of binned data
    for b in ['task_prestim', 'rest']:
        res = pd.read_pickle(HLTP_pupil.result_dir + '/LM_betas_full_random_' +#full_random_
                             b + '.pkl')
        df = pd.read_pickle(HLTP_pupil.result_dir +
                            '/roi_pwr_and_pupil_blocknorm_' + b + '.pkl')
        df.pupil = zscore(df.pupil)
        # bin data in pupil groups for presentation only
        group_percentile = np.arange(0., 100., 5)
        df['pupil_group'] =  np.digitize(df.pupil, np.percentile(df.pupil, group_percentile))
        mean_pupil_in_group = [np.mean(df.loc[df.pupil_group == g, "pupil"])
                for g in range(1, 21)]
        pup = np.arange(-2.5, 2.5, .1)

        colors = cm.winter(np.linspace(0, 255, 5).astype('int'))
        #roi_ylim = ([0.9, 1.5], [0.9, 1.5], [0.9, 1.5], [0.9, 1.5], 
         #           [0.9, 1.5], [0.9, 1.5], [0.9, 1.5])
        for roi in range(7):
            for f, fband in enumerate(HLTP_pupil.freq_bands.keys()):
                mpwr = np.zeros( (len(group_percentile) + 1, 24) )
                for sn, s in enumerate(HLTP_pupil.subjects):
                    subj_df = df.loc[df.subject == s]
                    subj_groups = np.unique(subj_df.pupil_group)
                    mpwr[subj_groups, sn] =  np.array(
                            [np.mean(subj_df.loc[subj_df.pupil_group == g, 
                                                 fband + str(roi)]) 
                        for g in subj_groups])
                m =  mpwr[1:].mean(axis = -1)
                e = mpwr[1:].std(axis = -1)/np.sqrt(24)
                fig, ax = plt.subplots(figsize = [1.5, 2.])

                plt.errorbar(mean_pupil_in_group, m, yerr = e,
                             alpha = .3, fmt = 'o', color = colors[f])
                
                if (res[fband + 'Qpval'][roi] < 0.05
                    ) & (res[fband + 'Qbic'][roi] < res[fband + 'Lbic'][roi]):
                    if np.isnan(res[fband + 'Q'][roi]):
                        res[fband + 'Q'][roi] = 0 # for plotting

                    plt.plot(pup, res[fband + 'inter'][roi] +
                             res[fband + 'Q'][roi] * pup **2 +
                            res[fband + 'L'][roi] * pup,
                            linewidth = 3, zorder = 100, color = colors[f],
                            label = fband)
                elif (res[fband + 'Lpval'][roi] < 0.05) & (
                        res[fband + 'Lbic'][roi] < res[fband + 'Qbic'][roi]):
                    plt.plot(pup, res[fband + 'inter'][roi] +
                             res[fband + 'L'][roi] * pup,
                             linewidth=3, zorder=100, color=colors[f],
                             label=fband)

                plt.ylim([(m - e).min()-.01, (m - e).min() + .29])
                plt.xlim([-2.5, 2.5])
                plt.locator_params(axis = 'y', nbins = 6)
                ax.spines['left'].set_position(('outward', 10))
                ax.yaxis.set_ticks_position('left')
                ax.spines['bottom'].set_position(('outward', 15))
                ax.xaxis.set_ticks_position('bottom')
                fig.savefig(figures_dir + '/roi_pwr_' + b + fband + str(roi) + 
                            '.png', bbox_inches = 'tight', transparent = True) 
                plt.close(fig)

def plot_fig_3B_random_effects():
    # Plot all the mixed model fit curves on top of binned data
    # this looks too messy / uninformative.
    model_x = np.arange(-5, 5, .1)
    colors = cm.winter(np.linspace(0, 1, 24))
    for b in ['task_prestim', 'rest']:
        df = pd.read_pickle(HLTP_pupil.result_dir +
                            '/roi_pwr_and_pupil_blocknorm_' + b + '.pkl')
        df.pupil = zscore(df.pupil)
        for roi in range(7):
            for f, fband in enumerate(HLTP_pupil.freq_bands.keys()):
                mdf_Q = pd.read_pickle(HLTP_pupil.result_dir +
                           '/LM_quadratic_full_rand_stats_' + str(roi)
                           + fband + b + '.pkl')
                mdf_L = pd.read_pickle(HLTP_pupil.result_dir +
                           '/LM_linear_full_rand_stats_' + str(roi)
                           + fband + b + '.pkl')
                s_params = np.array([mdf_Q.random_effects[s].values for s in HLTP_pupil.subjects])
                subj_order = np.flip(np.argsort(s_params[:, 0]))

                fig, ax = plt.subplots(figsize=[1.5, 2.])
                for color_id, subj_id in enumerate(subj_order):
                    subject = HLTP_pupil.subjects[subj_id]
                    plt.scatter(df[df.subject == subject].pupil.values,
                                df[df.subject == subject][fband + str(roi)].values, s=1,
                                alpha=0.1, color=colors[color_id])

                    if (mdf_Q.pvalues[1] < 0.05) & (mdf_Q.bic < mdf_L.bic):
                        params = mdf_Q.random_effects[subject]
                        group_params = mdf_Q.params
                        predicted_pwr = group_params[0] + (group_params[2] + params[1]) * model_x + \
                                 (group_params[1] + params[0]) * model_x**2

                    elif (mdf_L.pvalues[1] < 0.05) & (mdf_Q.bic > mdf_L.bic):
                        params = mdf_L.random_effects[subject]
                        group_params = mdf_L.params
                        predicted_pwr = group_params[0] + (group_params[1] + params[0]) * model_x
                    else: #no model is good
                        continue
                    plt.plot(model_x, predicted_pwr,
                                 alpha = 0.7, color = colors[color_id], linewidth = 2)

                plt.ylim([-2 , 2 ])
                plt.xlim([-5, 5])
                plt.locator_params(axis='y', nbins=6)
                ax.spines['left'].set_position(('outward', 10))
                ax.yaxis.set_ticks_position('left')
                ax.spines['bottom'].set_position(('outward', 15))
                ax.xaxis.set_ticks_position('bottom')
                fig.savefig(figures_dir + '/roi_pwr_randeff_' + b + fband + str(roi) +
                            '.png', bbox_inches='tight', transparent=True)
                plt.close(fig)

def plot_hmap(par_est, pvals, pvals_corr, lims = 0.03, savename = ''):
    fig, ax = plt.subplots(1, 1,figsize = (1.5,2))
    ax.imshow(~np.isnan(par_est), cmap = 'gray', alpha = 0.3)
    ax.imshow(par_est, cmap = 'Spectral_r', vmin = -lims, vmax = lims)

    for (j, i), _ in np.ndenumerate(pvals):
            if pvals_corr[j, i] < 0.05: # corrected for mc
                ax.text(i, j, '*', color = 'w', ha='center',va='center', fontsize = 13)
            elif pvals[j, i] < 0.05: # uncorrected
                ax.text(i, j, '+', color = 'w', ha='center',va='center', fontsize = 10)
    ax.axis('off')

    fig.savefig(figures_dir + '/' + savename + '.png',
                bbox_inches = 'tight', transparent = True)

def plot_fig_3C():

    n_rois = 7
    for b in ['task_prestim', 'rest']:
        res = pd.read_pickle(HLTP_pupil.result_dir + '/LM_betas_full_random_' + b + '.pkl')
        for mc in ['L', 'Q']:
            par_est = np.zeros((7,5)); pvals = np.zeros((7,5)); pvals_corr = np.zeros((7,5))
            for f, fband in enumerate(HLTP_pupil.freq_bands.keys()):
                for roi in range(n_rois):
                    par_est[roi, f] = res[fband + mc][roi]
                pvals[:, f] =  res[fband + mc + 'pval_corrected']
                pvals_corr[:, f] =  res[fband + mc + 'pval']
            if mc == 'L': lims = 0.05;
            else: lims = 0.03
            plot_hmap(par_est, pvals, pvals_corr, lims = lims, savename = 'roi_betas' + b + mc )

def plot_fig_3C_old():
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

plot_fig_3A('task_prestim_ds')
plot_fig_3A('mean_rest')
plot_fig_3B()
plot_fig_3C()






                                    