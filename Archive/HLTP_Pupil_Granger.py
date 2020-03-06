#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:05:53 2019

@author: podvae01
"""

# granger causality
from statsmodels.tsa.stattools import grangercausalitytests

x = np.zeros((2, len(all_pupil))) 
x[0, :] = all_pupil; x[1, :] = all_meg; # here in this example I take LCMV reconstructed 
res = grangercausalitytests(x.T, maxlag = 1)
res[1][0]['ssr_chi2test'][1]
x[1, :] = all_pupil; x[0, :] = all_meg;
res = grangercausalitytests(x.T, maxlag = 30)
res[1][0]['ssr_chi2test'][1]