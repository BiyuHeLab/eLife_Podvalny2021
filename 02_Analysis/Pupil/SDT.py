#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:05:03 2019

@author: podvae01
"""
import sys
sys.path.append('../../')
from os import path
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir, subjects
import numpy as np
import scipy
from scipy.optimize import curve_fit
import scipy.stats
from scipy.stats import sem
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import bic, aic

def sdt_from_df(df):
    n_rec_real = sum(df[df.real_img == True].recognition == 1)
    n_real = len(df[(df.real_img == True) & (df.recognition != 0)])
    n_rec_scr = sum(df[df.real_img == False].recognition == 1)
    n_scr = len(df[(df.real_img == False) & (df.recognition != 0)])
        
    HR, FAR, d, c = get_sdt_msr(n_rec_real, n_real, n_rec_scr, n_scr)
    return HR, FAR, d, c
        
def get_sdt_msr(n_rec_signal, n_signal, n_rec_noise, n_noise):
    Z = scipy.stats.norm.ppf
    if (n_noise == 0): FAR = np.nan
    else: FAR = max( float(n_rec_noise) / n_noise, 1. / (2 * n_noise) )
    if  n_signal == 0: HR = np.nan
    else: HR = min( float(n_rec_signal) / n_signal, 1 - (1. / (2 * n_signal) ) )
    d = Z(HR)- Z(FAR)
    c = -(Z(HR) + Z(FAR))/2.
    
    # return nans instead of infs
    if np.abs(d) == np.inf: d = np.nan
    if np.abs(c) == np.inf: c = np.nan
    return HR, FAR, d, c

def save_SDT_by_pupil():
    ''' calculate criterion and sensitivity in pupil groups '''
    bhv_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/all_subj_bhv_df_w_pupil.pkl')
    #group_percentile = np.arange(0., 100., 20);
    dfs = []
    for subject in subjects:
        p_group = bhv_df[bhv_df.subject == subject].pupil_size_pre
        bhv_group_val = {}
        
        for group in np.unique(p_group):
            group_df = bhv_df[(bhv_df.subject == subject) & 
                                     (bhv_df.pupil_size_pre == group)]
            
            HR, FAR, d, c = sdt_from_df(group_df)
            
            bhv_group_val['HR'] = [HR]; 
            bhv_group_val['FAR'] = [FAR]; 
            bhv_group_val['d'] = [d]; 
            bhv_group_val['c'] = [c];
            bhv_group_val['subject'] = subject; 
            bhv_group_val['group'] = [group]
            
            dfs.append(pd.DataFrame(bhv_group_val));
    
    sdt_df = pd.concat(dfs)
    sdt_df.to_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/sdt_by_pupil_df.pkl')
    
def stats_for_SDT_pupil():
    sdt_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/sdt_by_pupil_df.pkl')
    sdt_df = sdt_df.rename(columns={"group": "pupil"}) # for readability
    
    # fit a model for each behavioral variable
    for bhv_var in ['HR', 'FAR', 'c', 'd']:
        # quadratic
        mdf_Q = smf.mixedlm(bhv_var + " ~ np.power(pupil, 2) + pupil", sdt_df, 
                         groups = sdt_df["subject"]).fit()
        mdf_Q.bic = bic(mdf_Q.llf, mdf_Q.nobs, mdf_Q.df_modelwc)
        mdf_Q.save(HLTP_pupil.MEG_pro_dir +
                         '/results/mixedlmQ_pupil_' + bhv_var +'.pkl')
        #print(mdf_Q.summary())
        # linear
        mdf_L = smf.mixedlm(bhv_var + " ~ pupil", sdt_df,
                         groups = sdt_df["subject"]).fit()
        mdf_L.bic = bic(mdf_L.llf, mdf_L.nobs, mdf_L.df_modelwc)
        mdf_L.save(HLTP_pupil.MEG_pro_dir +
                         '/results/mixedlmL_pupil_' + bhv_var +'.pkl')
        #print(mdf_L.summary())
        # compare between two models (lower bic is better)
        if mdf_L.bic < mdf_Q.bic:
            print(bhv_var, "Linear term is better")
        else:  print(bhv_var, "Quadratic term is better")

def save_SDT_by_power(df_name, save_name, roi): 
    ''' calculate criterion and sensitivity in groups according to power''' 
    
    bhv_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                             '/results/' + df_name + '.pkl')
    
    group_percentile = np.arange(0., 100., 20);
    for fband, _ in HLTP_pupil.freq_bands.items():
        dfs = []
    
        for subject in subjects:
            # get power in each freq. band:
            pwr = bhv_df.loc[bhv_df.subject == subject, fband + roi]
            # cut to groups according to percentile:
            p_group = np.digitize(pwr, 
                                  np.percentile(pwr, group_percentile))
    
            bhv_group_val = {}
            
            for group in np.unique(p_group):
                group_df = bhv_df[bhv_df.subject == subject].loc[p_group == group]
                
                HR, FAR, d, c = sdt_from_df(group_df)
                
                bhv_group_val['HR'] = [HR]; 
                bhv_group_val['FAR'] = [FAR]; 
                bhv_group_val['d'] = [d]; 
                bhv_group_val['c'] = [c];
                bhv_group_val['subject'] = subject; 
                bhv_group_val['group'] = [group]
                
                dfs.append(pd.DataFrame(bhv_group_val));
        
        sdt_df = pd.concat(dfs)

        sdt_df.to_pickle(HLTP_pupil.MEG_pro_dir +
                                 '/results/' + save_name + roi + fband + '.pkl')

def stats_for_SDT_sensor_power(df_name, save_name):
    for band, _ in HLTP_pupil.freq_bands.items():
        sdt_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                             '/results/' + df_name + band + '.pkl')
        sdt_df = sdt_df.rename(columns={"group": "power"}) # for readability
        
        # fit a model for each behavioral variable
        for bhv_var in ['HR', 'FAR', 'c', 'd']:
            # quadratic
            
            mdf_Q = smf.mixedlm(bhv_var + " ~ np.power(power, 2) + power", 
                                sdt_df.dropna(), groups = sdt_df.dropna()["subject"]).fit()
            
            mdf_Q.bic = bic(mdf_Q.llf, mdf_Q.nobs, mdf_Q.df_modelwc)
            mdf_Q.save(HLTP_pupil.MEG_pro_dir + '/results/mixedlmQ_' + save_name
                       + band + bhv_var +'.pkl')
            #print(mdf_Q.summary())
            # linear
            mdf_L = smf.mixedlm(bhv_var + " ~ power", sdt_df.dropna(),
                             groups = sdt_df.dropna()["subject"]).fit()
            mdf_L.bic = bic(mdf_L.llf, mdf_L.nobs, mdf_L.df_modelwc)
            mdf_L.save(HLTP_pupil.MEG_pro_dir + '/results/mixedlmL_' + save_name
                       + band + bhv_var +'.pkl')
            #print(band, mdf_L.pvalues[1] < 0.05)
            # compare between two models (lower bic is better)
            if (mdf_Q.pvalues[1] < 0.05) | (mdf_Q.pvalues[2] < 0.05):
                print(save_name, band, bhv_var, mdf_Q.pvalues < 0.05)

def residual_power_behavior(fname, roi):
        bhv_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/' + fname + '.pkl')
        bhv_df = bhv_df[ ~((bhv_df.index == 288) & 
                                   (bhv_df.subject == 'BJB'))]
        b = 'task_prestim'
        group_percentile = np.arange(0., 100., 20);

        for fband, frange in HLTP_pupil.freq_bands.items():
            dfs = []
            for subject in HLTP_pupil.subjects:
                # get mean power spectrum
                pupil_states = HLTP_pupil.load(
                        HLTP_pupil.result_dir + '/pupil_states_' + 
                                    b + subject + '.pkl')
                pwr = bhv_df[bhv_df.subject == subject][fband + roi].values
                #if subject== 'BJB': pwr = pwr[:-1]

                dfs.append(pd.DataFrame({"subject" :subject, 
                    "power" :pwr, #np.log(
                    "pupil" :pupil_states.mean_pupil.values}))

            df = pd.concat(dfs)
            df.power = zscore(df.power)
            mdf_Q = smf.mixedlm("power ~ np.power(pupil, 2) + pupil", 
                            df.dropna(), groups = df.dropna()["subject"]
                            ).fit(method='powell')
            bhv_dfs = []
            for s, subject in enumerate(HLTP_pupil.subjects):
                power = df[df.subject == subject].power; 
                pupil = df[df.subject == subject].pupil; 
                res_pwr = (power -
                  mdf_Q.params[1] * (pupil ** 2) - 
                  mdf_Q.params[2] * (pupil) - 
                  mdf_Q.random_effects[subject].values[0])

                p_group = np.digitize(res_pwr, 
                                  np.percentile(res_pwr, group_percentile))
                
                bhv_group_val = {}
                
                for group in np.unique(p_group):
                    group_df = bhv_df[bhv_df.subject == subject
                                      ].loc[p_group == group]
                    
                    
                    HR, FAR, d, c = sdt_from_df(group_df)
                    
                    bhv_group_val['HR'] = [HR]; 
                    bhv_group_val['FAR'] = [FAR]; 
                    bhv_group_val['d'] = [d]; 
                    bhv_group_val['c'] = [c];
                    bhv_group_val['subject'] = subject; 
                    bhv_group_val['group'] = [group]
                    
                    bhv_dfs.append(pd.DataFrame(bhv_group_val));
            sdt_df = pd.concat(bhv_dfs)
            sdt_df.to_pickle(HLTP_pupil.MEG_pro_dir +
                                 '/results/SDT_res_pwr_' + fband + '.pkl')
            
def subj_fit_residual_power_behavior(fname, roi):
    bhv_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/' + fname + '.pkl')
    bhv_df = bhv_df[ ~((bhv_df.index == 288) & 
                                   (bhv_df.subject == 'BJB'))]
    b = 'task_prestim'
    group_percentile = np.arange(0., 100., 20);
    for fband, frange in HLTP_pupil.freq_bands.items():
        dfs = []
        for subject in HLTP_pupil.subjects:
            # get mean power spectrum
            pupil_states = HLTP_pupil.load(
                        HLTP_pupil.result_dir + '/pupil_states_' + 
                                    b + subject + '.pkl')
            pwr = bhv_df[bhv_df.subject == subject][fband + roi].values
            #if subject== 'BJB': pwr = pwr[:-1]

            dfs.append(pd.DataFrame({"subject" :subject, 
                    "power" :pwr, #np.log(
                    "pupil" :pupil_states.mean_pupil.values}))

        df = pd.concat(dfs)
        
        bhv_dfs = {'residual':[], 'pupil':[]}
        for s, subject in enumerate(HLTP_pupil.subjects):
            subj_df = df[df.subject == subject]
            subj_df.power = zscore(subj_df.power)
            mdf_Q = smf.mixedlm("power ~ np.power(pupil, 2) + pupil", 
                            subj_df.dropna(), groups = subj_df.dropna()["subject"]
                            ).fit(method='powell')
            pwr = {}
            pwr['residual'] = mdf_Q.resid.values
            pwr['pupil'] = (mdf_Q.params[1]*np.power(subj_df.pupil, 2) + 
                   mdf_Q.params[2]*subj_df.pupil)
            for k, res_pwr in pwr.items():
                p_group = np.digitize(res_pwr, 
                                  np.percentile(res_pwr, group_percentile))
                
                bhv_group_val = {}
                
                for group in np.unique(p_group):
                    group_df = bhv_df[bhv_df.subject == subject
                                      ].loc[p_group == group]
                    
                    
                    HR, FAR, d, c = sdt_from_df(group_df)
                    
                    bhv_group_val['HR'] = [HR]; 
                    bhv_group_val['FAR'] = [FAR]; 
                    bhv_group_val['d'] = [d]; 
                    bhv_group_val['c'] = [c];
                    bhv_group_val['subject'] = subject; 
                    bhv_group_val['group'] = [group]
                    
                    bhv_dfs[k].append(pd.DataFrame(bhv_group_val));
        for k in bhv_dfs.keys():
            sdt_df = pd.concat(bhv_dfs[k])
            sdt_df.to_pickle(HLTP_pupil.MEG_pro_dir +
                        '/results/SDT_res_pwr_subj_' + k + fband + '.pkl')           
            
# Run the functions
fname = 'all_subj_bhv_df_w_sensor_pwr'            
residual_power_behavior(fname)
stats_for_SDT_sensor_power('SDT_res_pwr_', 'res_power_')            

save_SDT_by_pupil();  stats_for_SDT_pupil() 
    
save_SDT_by_power('all_subj_bhv_df_w_sensor_pwr', 'sdt_by_PSD_df', '');  
stats_for_SDT_sensor_power('sdt_by_PSD_df', 'power_')

for roi in range(7):
    save_SDT_by_power('all_subj_bhv_df_w_roi_pwr', 'sdt_by_dics_df', str(roi));  
    stats_for_SDT_sensor_power('sdt_by_dics_df' + str(roi), 'dics_' + str(roi))

fname = 'all_subj_bhv_df_w_roi_pwr'  
for roi in range(7): 
    residual_power_behavior(fname,  str(roi))
    stats_for_SDT_sensor_power('SDT_res_pwr_', 'res_power_' + str(roi)) 
    
fname = 'all_subj_bhv_df_w_roi_pwr'  
for roi in range(7): 
    subj_fit_residual_power_behavior(fname,  str(roi))
    stats_for_SDT_sensor_power('SDT_res_pwr_subj_residual', 'res_power_' + str(roi))  
    stats_for_SDT_sensor_power('SDT_res_pwr_subj_pupil', 'pup_power_' + str(roi)) 
#------------------------------------------------------------------------------

## Same for source localized power  
#group_percentile = np.arange(0, 100, 20)    
#for band in ['l', 'm', 'h']:
#    for roi in range(7):
#        dfs = []
#        bhv_df['pwr_group'] = np.nan
#        
#        for subject in subjects:
#            pwr = bhv_df[(bhv_df.subject == subject)][band + '_power' + str(roi)]
#            perc = np.percentile(pwr, group_percentile)
#            p_group = np.digitize(pwr, perc)   
#            bhv_df.loc[bhv_df.subject == subject, 'pwr_group'] = p_group
#            bhv_group_val = {}
#            
#            for group in np.unique(p_group):
#                group_df = bhv_df[(bhv_df.subject == subject) & 
#                                         (bhv_df.pwr_group == group)]
#                
#                HR, FAR, d, c = sdt_from_df(group_df)
#                
#                bhv_group_val['HR'] = [HR]; 
#                bhv_group_val['FAR'] = [FAR]; 
#                bhv_group_val['d'] = [d]; 
#                bhv_group_val['c'] = [c];
#                bhv_group_val['subject'] = subject; 
#                bhv_group_val['group'] = [group]
#                
#                dfs.append(pd.DataFrame(bhv_group_val));
#        
#        sdt_df = pd.concat(dfs) 
#        savedir = figures_dir + '/bhv_roi' + str(roi) + 'freq' + band
#        make_figures(sdt_df, 'Power', savedir)
        
# calculate behavior as a 2d function of power and pupil groups
#
#for band in ['l', 'm', 'h']:
#    roi_mats = []
#    for roi in range(7):
#        bhv_mat = {'HR':[], 'FAR':[], 'd':[], 'c':[]}
#        for k in list(bhv_mat.keys()): 
#            bhv_mat[k] = np.zeros( ( len(pwr_groups), len(pup_groups)))
#        
#        for subject in subjects:
#            pwr = bhv_df[(bhv_df.subject == subject)][band + '_power' + str(roi)]
#            perc = np.percentile(pwr, group_percentile)
#            p_group = np.digitize(pwr, perc)   
#            bhv_df.loc[bhv_df.subject == subject, 'pwr_group'] = p_group
#        
#        pwr_groups = np.unique(bhv_df.pwr_group).astype(int)
#        pup_groups = np.unique(bhv_df.pupil_group).astype(int)
#        for pwr in pwr_groups:
#            for pup in pup_groups:
#                
#                HR = np.zeros((len(subjects)));FAR = np.zeros((len(subjects))); 
#                c = np.zeros((len(subjects))); d = np.zeros((len(subjects)));
#                
#                for s, subject in enumerate(subjects):
#                    
#                    group_df = bhv_df[(bhv_df.subject == subject) & 
#                                      (bhv_df.pwr_group == pwr) &
#                                      (bhv_df.pupil_group == pup)]
#                    
#                    HR[s], FAR[s], d[s], c[s] = sdt_from_df(group_df)
#                bhv_mat['HR'][pwr - 1, pup - 1] = np.nanmean(HR)
#                bhv_mat['FAR'][pwr - 1, pup - 1] = np.nanmean(FAR)
#                bhv_mat['c'][pwr - 1, pup - 1] = np.nanmean(c)
#                bhv_mat['d'][pwr - 1, pup - 1] = np.nanmean(d)
#
#        roi_mats.append(bhv_mat)
#        
#    m = np.array([roi_mats[i]['d'] for i in range(7)]).mean(axis = 0)
#    
#    fig, ax = plt.subplots(figsize = [2., 2.]);  
#    plt.imshow(m, cmap = 'hot', interpolation = 'hanning',
#               extent = [0, 100, 100,0], vmin = .2, vmax = .8)
#    
#    ax.grid(b = True, which = 'minor', color = 'w', linestyle = '--')
#    ax.grid(b = True, which = 'major', color = 'w', linestyle = '--')
#
#    ax.axis('on')
#    plt.box(True)
#    ax.spines['top'].set_visible(True);   
#    ax.spines['right'].set_visible(True)  
#    plt.xlabel('Pupil size (%)')
#    plt.ylabel('Power (%)')
#    ax.locator_params(axis='x', nbins=5); 
#    ax.locator_params(axis='y', nbins=5)
#    #plt.colorbar()
#    fig.savefig(figures_dir + '/bhv_heat_map_d_' 
#            + band + '.png', bbox_inches = 'tight', 
#            dpi = 800, transparent = True)    
#    
#    m = np.array([roi_mats[i]['c'] for i in range(7)]).mean(axis = 0)
#
#    fig, ax = plt.subplots(figsize = [2., 2.]);  
#    plt.imshow(m, cmap = 'hot', interpolation = 'hanning',
#               extent = [0, 100, 100,0], vmin = .2, vmax = .8)        
#    ax.grid(b = True, which = 'minor', color = 'w', linestyle = '--')
#    ax.grid(b = True, which = 'major', color = 'w', linestyle = '--')    
#    ax.axis('on'); plt.box(True)
#    ax.spines['top'].set_visible(True);   
#    ax.spines['right'].set_visible(True)  
#    plt.xlabel('Pupil size (%)')
#    plt.ylabel('Power (%)')
#    ax.locator_params(axis='x', nbins=5); 
#    ax.locator_params(axis='y', nbins=5)
#    fig.savefig(figures_dir + '/bhv_heat_map_c_' 
#            + band + '.png', bbox_inches = 'tight', 
#            dpi = 800, transparent = True)   
#
#    m = np.array([roi_mats[i]['HR'] for i in range(7)]).mean(axis = 0)
#    fig, ax = plt.subplots(figsize = [2., 2.]);  
#    plt.imshow(m, cmap = 'hot', interpolation = 'hanning',
#               extent = [0, 100, 100,0], vmin = .4, vmax = .6)        
#    ax.grid(b = True, which = 'minor', color = 'w', linestyle = '--')
#    ax.grid(b = True, which = 'major', color = 'w', linestyle = '--')    
#    ax.axis('on'); plt.box(True)
#    ax.spines['top'].set_visible(True);   
#    ax.spines['right'].set_visible(True)  
#    plt.xlabel('Pupil size (%)')
#    plt.ylabel('Power (%)')
#    ax.locator_params(axis='x', nbins=5); 
#    ax.locator_params(axis='y', nbins=5)
#    fig.savefig(figures_dir + '/bhv_heat_map_HR_' 
#            + band + '.png', bbox_inches = 'tight', 
#            dpi = 800, transparent = True)   
#    
#    m = np.array([roi_mats[i]['FAR'] for i in range(7)]).mean(axis = 0)
#
#    fig, ax = plt.subplots(figsize = [2., 2.]);  
#    plt.imshow(m, cmap = 'hot', interpolation = 'hanning',
#               extent = [0, 100, 100,0], vmin = .3, vmax = .5)        
#    ax.grid(b = True, which = 'minor', color = 'w', linestyle = '--')
#    ax.grid(b = True, which = 'major', color = 'w', linestyle = '--')    
#    ax.axis('on'); plt.box(True)
#    ax.spines['top'].set_visible(True);   
#    ax.spines['right'].set_visible(True)  
#    plt.xlabel('Pupil size (%)')
#    plt.ylabel('Power (%)')
#    ax.locator_params(axis='x', nbins=5); 
#    ax.locator_params(axis='y', nbins=5)
#    fig.savefig(figures_dir + '/bhv_heat_map_FAR_' 
#             + band + '.png', bbox_inches = 'tight', 
#            dpi = 800, transparent = True)   

