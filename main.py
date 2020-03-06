#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:39:47 2019

This doc explains the analysis order and other scirpts organization.

-------------------------------------------------------------------------------
1. MEG_preproc.py
-------------------------------------------------------------------------------
Preprocessing of MEG data. Running this script ends with generation of 
                       *_stage2_raw.fif 
containing ICA-cleaned contineous data for each experiment run

-------------------------------------------------------------------------------
2. Pupil_preproc.py
-------------------------------------------------------------------------------
This script preprocesses pupil size within each experiment block and 
generates files named: 
    clean_pupil_BlockName.pkl # replaced blinks with NaN
    clean_interp_pupil_BlockName.pkl # blinks are interpolated
    clean_interp_pupil_ds_BlockName.pkl # the above downsampled to 256 Hz
    
------------------------------------------------------------------------------- 
k.   
-------------------------------------------------------------------------------    
@author: podvae01
"""





# Analysis of pupil and behavior 

# Use this script to prepare epoched MEG and pupil data 
HLTP_Pupil_prepare_rest_epochs


# just a list of artifact components
HLTP_bad_ica_components

# sensor-level cross-correlation between pupil and brin
Sensor_pupil_cross_corr
