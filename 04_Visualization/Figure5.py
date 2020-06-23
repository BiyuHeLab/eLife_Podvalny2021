#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:16:59 2020
Behavior
@author: podvae01
"""
import fig_params
from fig_params import *
import HLTP_pupil
import numpy as np
#import mne
from matplotlib import pyplot as plt
import pandas as pd
figures_dir = HLTP_pupil.MEG_pro_dir  +'/_figures'

# plot
def plt_fig(group_percentile, m, e, ylimit, xlabel, ylabel):
    fig, ax = plt.subplots(1, 1,figsize = (1.5,2))
    plt.errorbar(group_percentile, m, e, color = 'gray', mec = 'k', linewidth = 1.5,
                 fmt='o', capsize = 5)
    plt.ylim(ylimit); plt.xlim([0, 100])
    plt.ylabel(ylabel, fontsize = 14)
    plt.xlabel(xlabel + ' (%)', fontsize = 14)
    ax.spines["top"].set_visible(False); 
    ax.spines["right"].set_visible(False) 
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    return fig, ax

def make_figures(sdt_df, xlabel, savedir, vtype):
    ylims = {'HR':[0.3, 0.6], 'FAR':[0., 0.3], 
             'c':[0.2, 1.],'d': [0.4, 1.2], 'p_corr':[0.45, 0.65],
             'recRT':[0.6, 0.8], 'catRT':[1, 1.2]}
    ylabels = {'HR':'Hit Rate', 'FAR':'FAR',
               'c':'criterion','d':'sensitivity', 'p_corr':'p correct',
               'catRT':'cat. RT', 'recRT':'rec. RT'}
    group_percentile = np.arange(10, 100, 20)
    for bhv_var in ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT', 'recRT']:
        mdf_Q = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir +
                             '/results/mixedlmQ_' + vtype + bhv_var +'.pkl')
        print(mdf_Q.pvalues, mdf_Q.params)
        mdf_L = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir +
                             '/results/mixedlmL_' + vtype + bhv_var +'.pkl')
        data = np.array([sdt_df[sdt_df.group == grp][bhv_var] 
                                        for grp in sdt_df.group.unique()])
        m = np.nanmean(data, axis = -1);
        e = np.nanstd(data, axis = -1) / np.sqrt(24)    
        fig, ax = plt_fig(group_percentile, m, e, ylims[bhv_var], xlabel, 
                          ylabels[bhv_var])
        if (mdf_Q.pvalues[1] < 0.05) | (mdf_Q.pvalues[2] < 0.05):
            plt.plot(np.arange(1, 100),
                     mdf_Q.params[0] + mdf_Q.params[1] * np.linspace(1, 5, 99) ** 2
                     + mdf_Q.params[2] * np.linspace(1, 5, 99),
                     color = 'k', linewidth = 2)
           
        if (mdf_L.pvalues[1] < 0.05):
             plt.plot(np.arange(1, 100),
                     mdf_L.params[0] + mdf_L.params[1] * np.linspace(1, 5, 99),
                     color = 'gray', linewidth = 2)
        fig.savefig(savedir + '_' + bhv_var + '.png', dpi = 800, 
                    bbox_inches = 'tight', transparent=True)
    
def plot_betas(mod_file, savedir, ylims):
    #ylims = {'HR':[-0.04, 0.04], 'FAR':[-0.04, 0.04], 'c':[-0.05, 0.05],'d': [-0.1, 0.1]}

    for bhv_var in ['HR', 'FAR', 'c', 'd']:
        errs = []; betas = []
        for fband, frange in HLTP_pupil.freq_bands.items():

            #mdf_Q = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir +
            #                     '/results/mixedlmQ_' + mod_file + fband + bhv_var +'.pkl')
            #if mdf_Q.pvalues[1] < 0.05:
            #    print(bhv_var, fband, mdf_Q.pvalues[1])
            mdf_Q = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir +
                                 '/results/mixedlmL_' 
                                 + mod_file + fband + bhv_var +'.pkl')
            if mdf_Q.pvalues[1] < 0.05:
                print(bhv_var, fband, mdf_Q.pvalues[1])
            betas.append(mdf_Q.params[1])
            errs.append(mdf_Q.bse[1])
        fig, ax = plt.subplots(1, 1,figsize = (1.5,2))     
        plt.bar([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$'], 
                betas, yerr = errs, facecolor = 'w', edgecolor = 'k', capsize = 4)
        plt.ylim(ylims[bhv_var])
        ax.spines["top"].set_visible(False); 
        ax.spines["right"].set_visible(False) 
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        plt.ylabel('', fontsize = 14)
        plt.xlabel('Freq. band', fontsize = 14)
        fig.savefig(savedir + '_betas_' + bhv_var + '.png', dpi = 800, 
                    bbox_inches = 'tight', transparent=True)

def plot_res_power_BHV(savedir):
    ylims = {'HR':0.02, 'FAR':0.02, 
             'c':0.05,'d': 0.05, 'p_corr':0.02, 'catRT':0.02, 'recRT':0.02}
    ylabels = {'HR':'Hit Rate', 'FAR':'FAR',
               'c':'criterion','d':'sensitivity', 'p_corr':'p correct',
               'catRT':'cat. RT', 'recRT':'rec. RT'}
    #group_percentile = np.arange(10, 100, 20)
    for bhv_var in ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT', 'recRT']:        
        [table_par_e, table_pvals] = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + 
                        '/results/respe_pval_' + bhv_var +'.pkl')
        
        fig, ax = plt.subplots(1, 1,figsize = (1.5,2))     
        plt.imshow(table_par_e, vmin = -ylims[bhv_var], vmax = ylims[bhv_var], 
                   cmap = 'Spectral_r')
        ax.axis('off')
        #plt.box(True)
        #ax.spines['top'].set_visible(True);    ax.spines['right'].set_visible(True)
        fig.savefig(savedir + '_betas_' + bhv_var + '.png', dpi = 800, 
                    bbox_inches = 'tight', transparent=True)
        
   
def plot_cat_decoding(savedir):
    scores = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + 
                        '/results/cat_decod_score.pkl')       
    fig, ax = plt.subplots(1, 1,figsize = (3, 5))     
    plt.imshow(scores.mean(axis = -1), vmin = 0.2, vmax = 0.3,
                       cmap = 'Spectral_r')    
    ax.axis('off')
    fig.savefig(savedir + '_cat_decode_heatmap.png', dpi = 800, 
                    bbox_inches = 'tight', transparent=True)
    
    m = scores.mean(axis = 0).mean(axis = -1)
    e = scores.mean(axis = 0).std(axis = -1) / np.sqrt(24)
    fig, ax = plt.subplots(1, 1,figsize = (7, 2))     
    plt.errorbar(np.arange(0, 2., .1)+.05, m, e, color = 'gray', mec = 'k', linewidth = 1.5,
                 fmt='--', capsize = 5)
    plt.plot([0,2], [0.25, 0.25], 'r:')
    plt.ylim([0.22, 0.3]); plt.xlim([0, 2])
    ax.spines["top"].set_visible(False); 
    ax.spines["right"].set_visible(False) 
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Time (s)', fontsize = 14)
    fig.savefig(savedir + '_cat_decode_time.png', dpi = 800, 
                    bbox_inches = 'tight', transparent=True)
    m = scores.mean(axis = 1).mean(axis = -1)
    e = scores.mean(axis = 1).std(axis = -1) / np.sqrt(24)
    
    fig, ax = plt.subplots(1, 1,figsize = (2, 2))     
    plt.errorbar(np.arange(0, 100, 20)+10, m, e, color = 'gray', mec = 'k', linewidth = 1.5,
                 fmt='--', capsize = 5)
    plt.ylim([0.22, 0.3]); plt.xlim([0, 100])
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Pupil (%)', fontsize = 14)
    ax.spines["top"].set_visible(False); 
    ax.spines["right"].set_visible(False) 
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    plt.plot([0, 100], [0.25, 0.25], 'r:')
    fig.savefig(savedir + '_cat_decode_pupil.png', dpi = 800, 
                    bbox_inches = 'tight', transparent=True) 

# FIGURE Pupil -> Behavior   

#IV = 'pupil_size_pre'
IV = 'pupil'
sdt_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir + 
                        '/results/sdt_by_' + IV + '.pkl')
savedir = figures_dir + '/bhv_pupil_'    
make_figures(sdt_df, 'Pupil ', savedir, IV + '_' )     


sdt_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/sdt_by_pupil_resid_df.pkl')        
make_figures(sdt_df, 'Res Pupil', figures_dir + '/bhv_pupil_res', 'pupi_res_')

# FIGURE Pupil -> Stimulus-triggered response   
plot_cat_decoding(figures_dir + '/stim'  )

# FIGURE Power -> Behavior

    
plot_res_power_BHV(savedir)



# ylims = {'HR':[-0.04, 0.04], 'FAR':[-0.04, 0.04], 
#          'c':[-0.1, 0.1],'d': [-0.1, 0.1]}
# for roi in range(7):
#     print('roi', str(roi))
#     savedir = figures_dir + '/bhv_res_roi_pwrL' + str(roi)   
#     plot_betas('res_power_' + str(roi), savedir, ylims)

# # FIGURE 4B Sensor-level power -> Behavior   
# savedir = figures_dir + '/bhv_pwr'    
# for fband, frange in HLTP_pupil.freq_bands.items():
#     sdt_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
#                              '/results/SDT_res_pwr_subj_residual' + fband + '.pkl')
#     make_figures(sdt_df, 'Power ', savedir + fband, 'res_power_' + fband)           

# savedir = figures_dir + '/bhv_pwr'    

# plot_betas('power_', savedir)   

# # FIGURE 4C Source-level power -> Behavior  

    
    
# ylims = {'HR':[-0.1, 0.1], 'FAR':[-0.1, 0.1], 'c':[-0.5, 0.5],'d': [-0.5, 0.5]}

# for roi in range(7):
#     savedir = figures_dir + '/bhv_pwr_dicsQL' + str(roi)   
#     print('roi', str(roi))
#     plot_betas('dics_' + str(roi), savedir, ylims)



# # residual analysis
# savedir = figures_dir + '/bhv_res_pwr'   
    
# for fband, frange in HLTP_pupil.freq_bands.items():
#     sdt_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
#                              '/results/SDT_res_pwr_' + fband + '.pkl')
#     make_figures(sdt_df, 'Power ', savedir + fband, 'res_power_' + fband)
# savedir = figures_dir + '/bhv_res_pwrQL'
# plot_betas('res_power_', savedir)
