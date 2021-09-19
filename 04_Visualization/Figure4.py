#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:16:59 2020
Figure 4, run the following scripts to get the results:
Pupil/03_SDT.py

@author: podvae01
"""
import fig_params
from fig_params import *
import HLTP_pupil
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore

figures_dir = HLTP_pupil.MEG_pro_dir  +'/_figures'
# plot
def plt_fig(group_percentile, m, e, ylimit, xlabel, ylabel):
    """Error bar plot"""
    fig, ax = plt.subplots(1, 1,figsize = (1.5,2))
    plt.errorbar(group_percentile, m, e, color = 'gray', mec = 'k', linewidth = 1.5,
                 fmt='o', capsize = 5)
    plt.ylim(ylimit); plt.xlim([-2.12132034, 2.12132034])
    plt.ylabel(ylabel, fontsize = 14)
    plt.xlabel(xlabel, fontsize = 14)
    ax.spines["top"].set_visible(False); 
    ax.spines["right"].set_visible(False) 
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    return fig, ax

def make_figures(sdt_df, xlabel, savedir, vtype):
    """plot the model fit on top of data error bar plot"""
    ylims = {'HR':[0.3, 0.6], 'FAR':[0., 0.3], 
             'c':[0.2, 1.],'d': [0.4, 1.2], 'p_corr':[0.45, 0.65],
             'catRT':[1, 1.2]}
    ylabels = {'HR':'Hit Rate', 'FAR':'FAR',
               'c':'criterion','d':'sensitivity', 'p_corr':'p correct',
               'catRT':'cat. RT'}
    group_bin = zscore(np.arange(1, 6))
    model_x = (np.arange(0, 6, .1) - np.arange(1, 6).mean()) / np.arange(1, 6).std()

    for bhv_var in ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT']:
        mdf_Q = HLTP_pupil.load(HLTP_pupil.result_dir +
                             '/mixedlmQ_' + vtype + bhv_var + '.pkl')
        print("Quadraic model ", mdf_Q.summary(), mdf_Q.aic)
        mdf_L = HLTP_pupil.load(HLTP_pupil.result_dir +
                             '/mixedlmL_' + vtype + bhv_var +'.pkl')
        print("Linear model ", mdf_L.summary(), mdf_L.aic)

        data = np.array([sdt_df[sdt_df.group == grp][bhv_var] 
                                        for grp in sdt_df.group.unique()])
        m = np.nanmean(data, axis = -1)
        e = np.nanstd(data, axis = -1) / np.sqrt(24)    

        fig, ax = plt_fig(group_bin, m, e, ylims[bhv_var], xlabel,
                          ylabels[bhv_var])

        if (mdf_Q.pvalues[1] < 0.05):
            plt.plot(model_x,
                     mdf_Q.params[0] + mdf_Q.params[1] * model_x ** 2
                     + mdf_Q.params[2] * model_x,
                     color = 'k', linewidth = 2)
        if (mdf_L.pvalues[1] < 0.05):
             plt.plot(model_x,
                     mdf_L.params[0] + mdf_L.params[1] * model_x,
                     color = 'gray', linewidth = 2)
        ax.set_xticks((np.arange(0, 7) - np.arange(1, 6).mean()) / np.arange(1, 6).std())
        ax.set_xticklabels(['', '1', '2', '3', '4', '5', ''])
        fig.savefig(savedir + '_' + bhv_var + '.png', dpi = 800, 
                    bbox_inches = 'tight', transparent=True)

def plot_random_effects(sdt_df, xlabel, savedir, vtype):
    """plot the model fit on top of data error bar plot"""
    ylims = {'HR':[0., 1.], 'FAR':[0., 1.],
             'c':[-1., 2.],'d': [-1.5, 2.5], 'p_corr':[0.2, 0.8],
             'recRT':[0.2, 1.2], 'catRT':[0.5, 1.5]}
    ylabels = {'HR':'Hit Rate', 'FAR':'FAR',
               'c':'criterion','d':'sensitivity', 'p_corr':'p correct',
               'catRT':'cat. RT', 'recRT':'rec. RT'}
    colors = cm.winter(np.linspace(0, 1, 24))
    group_bin = zscore(np.arange(1, 6))
    model_x = (np.arange(0, 6, .1)- np.arange(1, 6).mean()) / np.arange(1, 6).std()
    for bhv_var in ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT', 'recRT']:

        mdf_L = HLTP_pupil.load(HLTP_pupil.result_dir +
                             '/mixedlmL_' + vtype + bhv_var +'.pkl')
        mdf_Q = HLTP_pupil.load(HLTP_pupil.result_dir +
                             '/mixedlmQ_' + vtype + bhv_var +'.pkl')

        s_params = np.array([mdf_Q.random_effects[s].values for s in HLTP_pupil.subjects])
        subj_order = np.argsort(s_params[:, 0])
        fig, ax = plt.subplots(1, 1,figsize = (1.5,2))
        plt.ylim(ylims[bhv_var]); plt.xlim([-2.12132034, 2.12132034])
        plt.ylabel(bhv_var, fontsize = 14)
        plt.xlabel(xlabel, fontsize = 14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks((np.arange(0, 7) - np.arange(1, 6).mean()) / np.arange(1, 6).std())
        ax.set_xticklabels(['', '1', '2', '3' , '4', '5', ''])
        for color_id, subj_id in enumerate(subj_order):
            subject = HLTP_pupil.subjects[subj_id]
            subj_data = sdt_df[sdt_df.subject == subject][bhv_var].values
            plt.plot(group_bin, subj_data, 'o', markeredgecolor = 'None', markerfacecolor = colors[color_id],
                         alpha = 0.5)
            if (mdf_L.pvalues[1] < 0.05) & (mdf_L.bic < mdf_Q.bic):
                #plt.plot(zscore(sdt_df.group), mdf_L.fittedvalues, '.')
                params = mdf_L.random_effects[subject]
                group_params = mdf_L.params
                plt.plot(model_x, params[0] + group_params[0] + (group_params[1] + params[1]) * model_x, alpha = 0.5,
                     color = colors[color_id], linewidth = 2)
            elif (mdf_Q.pvalues[1] < 0.05) & (mdf_Q.bic < mdf_L.bic):
                params = mdf_Q.random_effects[subject]
                group_params = mdf_Q.fe_params
                plt.plot(model_x, (group_params[0] + params[0]) + (group_params[2] + params[2]) * model_x
                             + (group_params[1] + params[1]) * (model_x**2), alpha = 0.5,
                        color = colors[color_id], linewidth = 2)


        fig.savefig(savedir + '_re_' + bhv_var + '.png', dpi = 800,
                    bbox_inches = 'tight', transparent=True)

def plot_betas(mod_file, savedir, ylims):
    """bar plot of model weights"""
    #ylims = {'HR':[-0.04, 0.04], 'FAR':[-0.04, 0.04], 'c':[-0.05, 0.05],'d': [-0.1, 0.1]}

    for bhv_var in ['HR', 'FAR', 'c', 'd']:
        errs = []; betas = []
        for fband, frange in HLTP_pupil.freq_bands.items():

            #mdf_Q = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir +
            #                     '/results/mixedlmQ_' + mod_file + fband + bhv_var +'.pkl')
            #if mdf_Q.pvalues[1] < 0.05:
            #    print(bhv_var, fband, mdf_Q.pvalues[1])
            mdf_Q = HLTP_pupil.load(HLTP_pupil.result_dir +
                                 '/mixedlmL_'
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
def plot_cat_decoding(savedir):

    scores, perm_scores = HLTP_pupil.load(HLTP_pupil.result_dir +
                        '/cat_decod_score.pkl')
    fig, ax = plt.subplots(1, 1,figsize = (3, 5))
    plt.imshow(scores.mean(axis = -1), vmin = 0.2, vmax = 0.3,
                       cmap = 'Spectral_r')
    ax.axis('off')
    fig.savefig(savedir + '_cat_decode_heatmap.png', dpi = 800,
                    bbox_inches = 'tight', transparent=True)

    m = scores.mean(axis = 0).mean(axis = -1)
    e = scores.mean(axis = 0).std(axis = -1) / np.sqrt(24)

    p_val = np.ones(20)
    mean_perm_score = perm_scores.mean(axis=0).mean(axis=1)
    for t in range(20):
        p_val[t] = (mean_perm_score[t, :] > m[t]).mean()
    corr_pval = multipletests(p_val, alpha=0.05, method='fdr_bh')
    fig, ax = plt.subplots(1, 1,figsize = (7, 2))

    plt.errorbar(np.arange(0, 2., .1)+.05, m, e, color = 'gray', mec = 'k', linewidth = 1.5,
                 fmt='--', capsize = 5)
    plt.plot(np.arange(0, 2., .1)+.05, mean_perm_score.mean(axis = -1), 'r:')
    plt.plot(np.arange(0, 2., .1)+.05, 0.295 * corr_pval[0], 'k*')
    plt.plot()
    plt.ylim([0.22, 0.3]); plt.xlim([0, 2])
    ax.spines["top"].set_visible(False)
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
    p_val = np.ones(5)
    mean_perm_score = perm_scores.mean(axis=1).mean(axis=1)
    mean_score = scores.mean(axis=1).mean(axis=-1)
    for t in range(5):
        p_val[t] = (mean_perm_score[t, :] > mean_score[t]).mean()
    corr_pval = multipletests(p_val, alpha=0.05, method='fdr_bh')

    fig, ax = plt.subplots(1, 1,figsize = (2, 2))
    plt.errorbar(y = np.arange(0, 100, 20)+10, x = np.flip(m), xerr= np.flip(e),
                 color = 'gray', mec = 'k', linewidth = 1.5,
                 fmt='--', capsize = 5)
    plt.xlim([0.24, 0.3]); plt.ylim([0, 100])
    plt.xlabel('Accuracy', fontsize = 14)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    plt.plot(np.flip(mean_perm_score.mean(axis = -1)), np.arange(0, 100, 20) + 10, 'r--')
    plt.plot(np.flip(0.295 * corr_pval[0]), np.arange(0, 100, 20) + 10, 'k*')

    fig.savefig(savedir + '_cat_decode_pupil.png', dpi = 800,
                    bbox_inches = 'tight', transparent=True)

def plot_cat_decoding_suppl(savedir):
    scores, perm_scores = HLTP_pupil.load(HLTP_pupil.result_dir +
                                          '/cat_decod_score.pkl')
    m = scores.mean(axis=0).mean(axis=-1)
    e = scores.mean(axis=0).std(axis=-1) / np.sqrt(24)

    p_val = np.ones(20)
    mean_perm_score = perm_scores.mean(axis=0).mean(axis=1)
    for t in range(20):
        p_val[t] = (mean_perm_score[t, :] > m[t]).mean()
    corr_pval = multipletests(p_val, alpha=0.05, method='fdr_bh')
    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
    plt.scatter(np.tile(np.expand_dims(np.arange(0, 2., .1), axis=1), 24) + .05,
                scores.mean(axis=0), s=15, alpha=0.1, color = 'k')
    plt.errorbar(np.arange(0, 2., .1) + .05, m, e, color='gray', mec='k', linewidth=1.5,
                 fmt='--', capsize=5)
    plt.plot(np.arange(0, 2., .1) + .05, mean_perm_score.mean(axis=-1), 'r:')
    plt.plot(np.arange(0, 2., .1)+.05, 0.39 * corr_pval[0], 'k*')
    plt.ylim([0.1, 0.4])
    plt.xlim([0, 2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    fig.savefig(savedir + '_suppl_cat_decode_time.png', dpi=800,
                bbox_inches='tight', transparent=True)
    m = scores.mean(axis=1).mean(axis=-1)
    e = scores.mean(axis=1).std(axis=-1) / np.sqrt(24)
    p_val = np.ones(5)
    mean_perm_score = perm_scores.mean(axis=1).mean(axis=1)
    mean_score = scores.mean(axis=1).mean(axis=-1)
    for t in range(5):
        p_val[t] = (mean_perm_score[t, :] > mean_score[t]).mean()
    corr_pval = multipletests(p_val, alpha=0.05, method='fdr_bh')

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    plt.scatter(
                np.tile(np.expand_dims(np.arange(1, 6), axis=1), 24),scores.mean(axis=1),
        s = 15, alpha = 0.1, color = 'k')
    plt.errorbar(x = np.arange(1, 6), y = m, yerr = e,
                 color = 'gray', mec = 'k', linewidth = 1.5,
                 fmt = '--', capsize =5 )
    plt.ylim([0.1, 0.4])
    plt.xlim([0, 6])
    plt.ylabel('Accuracy', fontsize=14)
    plt.plot(np.arange(1, 6), mean_perm_score.mean(axis=-1), 'r--')
    plt.plot(np.arange(1, 6), 0.39 * corr_pval[0], 'k*')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, 7) )

    ax.set_xticklabels(['', '1', '2', '3', '4', '5', ''])

    fig.savefig(savedir + '_suppl_cat_decode_pupil.png', dpi=800,
                bbox_inches='tight', transparent=True)


# Plot Figure 4C-D
IV = 'pupil'
sdt_df = pd.read_pickle(HLTP_pupil.result_dir +
                        '/sdt_by_' + IV + '.pkl')
savedir = figures_dir + '/bhv_pupil_'    
make_figures(sdt_df, 'Pupil bin', savedir, IV + '_' )

# FIGURE Pupil -> Stimulus-triggered response
plot_cat_decoding(figures_dir + '/stim'  )
plot_cat_decoding_suppl(figures_dir + '/stim' )



