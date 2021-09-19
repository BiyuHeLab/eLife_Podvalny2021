#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:39:47 2019

This doc explains the analysis order and other scripts organization.
@author: podvae01
"""
#-------------------------------------------------------------------------------
# 1. Preprocessing
#-------------------------------------------------------------------------------
# All files are located in a folder named 01_Preprocessing.

# Behavior: read the raw psychtoolbox matlab files and organize in a dataframe
01_prepare_behavior_dataframe

# Pupil: preprocesses pupil size within each experiment block and
# generate files named:
#    clean_pupil_BlockName.pkl # replaced blinks with NaN
#    clean_interp_pupil_BlockName.pkl # blinks are interpolated
#    clean_interp_pupil_ds_BlockName.pkl # the above downsampled to 256 Hz
02_pupil_preproc

# MEG data: Running this script ends with generation of *_stage2_raw.fif
# files containing ICA-cleaned continuous data for each experiment run
# also prepares epochs
03_MEG_preproc

#-------------------------------------------------------------------------------
# 2. Analysis
#-------------------------------------------------------------------------------
01_pupil_timecourse_analysis
02_pupil_fast_events
03_SDT
# sensor-level cross-correlation between pupil and brain
Sensor_pupil_cross_corr
