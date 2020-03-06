#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:22:17 2018

Read matlab files that store Psychtoolbox experiment data and translate
the behavioral data of all subjects to pandas data frame. This preprocessing 
step is needed to have a more convenient access to the behavioral data

@author: podvae01
"""
import numpy as np
from scipy.io import loadmat
import pandas as pd
import HLTP

def PTmat_files_to_dict(subject):
    '''
    create a python dictionary from Psychtoolbox MATLAB files for each subject
    '''
    bhv_dict = {}
    data_dir = HLTP.MEG_raw_dir + '/' + subject
    bhv = loadmat(data_dir  + '/Behavior/bhv_data.mat')['bhv_data']
    matfields = ['seen', 'unseen', 'real_img']
    for m in matfields:
        bhv_dict[m] = np.squeeze(bhv[m][0,0][0]).astype('bool')
    n = len(bhv_dict[m])    
    bhv_dict['cat_protocol'] = np.squeeze(bhv['cat_protocol'][0,0][0])[:n] 
    bhv_dict['cat_response'] = np.squeeze(bhv['cat_response'][0,0][0])[:n] 
    
    bhv = loadmat(data_dir +'/Behavior/datafile.mat')['data']
    bhv_dict['stimID'] = bhv['stimID'][0,0][0][:n] 
    bhv_dict['exemplar'] = [i.astype('str')[0] for i in 
            bhv['exemplar'][0,0][0]][:n]    
    return bhv_dict

def PTmat_quest_files_to_dict(subject):
    bhv_dict = {}
    data_dir = HLTP.MEG_raw_dir + '/' + subject
    bhv = loadmat(data_dir  + '/Behavior/bhv_dataQ.mat')['bhv_dataQ']
    bhv_dict['recognition'] = np.squeeze(bhv['seen'][0,0][0]).astype('int') - \
        np.squeeze(bhv['unseen'][0,0][0]).astype('int')
    bhv_dict['contrast'] = np.squeeze(bhv['contrast'][0,0][0]).astype('float')         
    return bhv_dict        
        
        
        
# place the dictionary in a data frame for each subject and remove bad blink 
# trials
df_list = []
quest_df_list = []
for subject in HLTP.subjects:
    bhv_df = pd.DataFrame(PTmat_files_to_dict(subject))
    bhv_df['subject'] = subject    
    bhv_df['correct'] = bhv_df['cat_protocol'] == bhv_df['cat_response']
    try:
        bad_trials = HLTP.load(HLTP.MEG_pro_dir + '/' + subject + '/bad_trials.p')
    except:
        print('\n Identify bad_trials based on blinks first. '
              '\n Use HLTP_eye_tracker.py script')
        break;

    df_list.append(bhv_df)
    
    quest_df = pd.DataFrame(PTmat_quest_files_to_dict(subject))
    quest_df['subject'] = subject   
    quest_df_list.append(quest_df)
    
all_df = pd.concat(df_list)
all_df['recognition'] = 1*all_df['seen'] - 1*all_df['unseen']
all_df.to_pickle(HLTP.MEG_pro_dir +'/results/all_subj_bhv_df.pkl')

all_df = pd.concat(quest_df_list)
all_df.to_pickle(HLTP.MEG_pro_dir +'/results/all_quest_df.pkl')






