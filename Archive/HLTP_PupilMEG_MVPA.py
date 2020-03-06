#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:15:26 2018

@author: podvae01
"""
import HLTP
from mne.decoding import LinearModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate

model = make_pipeline(StandardScaler(), 
                      LinearModel(LogisticRegression(C = 1))) 
cv = KFold(n_splits = 5)   #
scoring = 'roc_auc'        #Classifier scoring method

data, labels = load_pupil_meg_corr_data() #TODO

all_scores = []
for sub_idx, _ in enumerate(HLTP.subjects):
    cv_result = cross_validate(model, data[sub_idx], labels[sub_idx], 
                               cv = cv, scoring = scoring)
    all_scores.append(cv_result['test_score'].mean())
    
#ADD another loop for more time points