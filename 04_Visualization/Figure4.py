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
    ylims = {'HR':[0.3, 0.6], 'FAR':[0., 0.3], 'c':[0.2, 1.],'d': [0.4, 1.2]}
    ylabels = {'HR':'Hit Rate', 'FAR':'FAR', 'c':'criterion','d':'sensitivity'}
    group_percentile = np.arange(10, 100, 20)
    for bhv_var in ['HR', 'FAR', 'c', 'd']:
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
                                 '/results/mixedlmL_' + mod_file + fband + bhv_var +'.pkl')
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
        
# FIGURE 4A Pupil -> Behavior   
sdt_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir + 
                        '/results/sdt_by_pupil_df.pkl')
savedir = figures_dir + '/bhv_pupil'    
make_figures(sdt_df, 'Pupil ', savedir, 'pupil_')           

# FIGURE 4B Sensor-level power -> Behavior   
savedir = figures_dir + '/bhv_pwr'    
for fband, frange in HLTP_pupil.freq_bands.items():
    sdt_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                             '/results/sdt_by_PSD_df' + fband + '.pkl')
    make_figures(sdt_df, 'Power ', savedir + fband, 'power_' + fband)           

savedir = figures_dir + '/bhv_pwr'    

plot_betas('power_', savedir)   

# FIGURE 4C Source-level power -> Behavior  
ylims = {'HR':[-0.1, 0.1], 'FAR':[-0.1, 0.1], 'c':[-0.5, 0.5],'d': [-0.5, 0.5]}

for roi in range(7):
    savedir = figures_dir + '/bhv_pwr_dicsQL' + str(roi)   
    print('roi', str(roi))
    plot_betas('dics_' + str(roi), savedir, ylims)

ylims = {'HR':[-0.04, 0.04], 'FAR':[-0.04, 0.04], 'c':[-0.1, 0.1],'d': [-0.1, 0.1]}
for roi in range(7):
    print('roi', str(roi))
    savedir = figures_dir + '/bhv_res_roi_pwrL' + str(roi)   
    plot_betas('res_power_' + str(roi), savedir, ylims)

# residual analysis
savedir = figures_dir + '/bhv_res_pwr'   
    
for fband, frange in HLTP_pupil.freq_bands.items():
    sdt_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                             '/results/SDT_res_pwr_' + fband + '.pkl')
    make_figures(sdt_df, 'Power ', savedir + fband, 'res_power_' + fband)
savedir = figures_dir + '/bhv_res_pwrQL'
plot_betas('res_power_', savedir)
