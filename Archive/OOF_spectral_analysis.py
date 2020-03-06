#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:10:31 2019

OOF = One Over F

@author: podvae01
"""
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import HLTP
import mne
import scipy

def get_blinks(pupil_data):
    thr = -3.8;             
    pupil_data[0] = thr;
    blinks = np.diff((pupil_data < thr).astype(int), n = 1, axis = 0)
    blink_start = (blinks == 1).nonzero()[0]
    blink_end = (blinks == -1).nonzero()[0]
    
    if len(blink_start) > len(blink_end):
        blink_end=np.append(blink_end, len(pupil_data))
    return blink_start, blink_end

# TODO: Do here better job on cleaning up the blinks edges
def clean_pupil(pupil_data):
    blink_start, blink_end = get_blinks(pupil_data)
    tt = 600 * 0.1
    for ii in range(len(blink_start)):
        pupil_data[blink_start[ii]: blink_end[ii]] = np.NaN
        s =  int(blink_start[ii] - tt); e = int(blink_end[ii] + tt)
        if ( s >= 0) & ( e < len(pupil_data)):
            pupil_data[ s: e] = np.NaN
    
    time = np.arange(len(pupil_data)); nnan_ind = ~np.isnan(pupil_data)
    m, b, r_val, p_val, std_err = scipy.stats.linregress(time[nnan_ind], pupil_data[nnan_ind])
    pupil_data = pupil_data - (m*time + b)
    return pupil_data
    
def oof(freq, exp, A):
    '''Define the one over F function'''
    return A + (exp * freq)



def get_oof_param(fractal, freq, freq_range):
    '''fit the oof to data and get parameters'''
    samples = (freq > freq_range[0]) & (freq < freq_range[1])
    param, _ = curve_fit(oof, np.log(freq[samples]), np.log(fractal[samples]))
    exponent = param[0]; amplitude = np.exp(param[1]);
    return exponent, amplitude

def padding(xx, ntimes):
    '''padding for spectral analysis'''
    xxp = np.zeros([ntimes * len(xx)])
    xxp[:len(xx)] = xx - xx.mean(axis = 0)
    return xxp

def get_spectral_components(data, sampling_frequency, N_windows):
    '''estimate fractal and harmonic spectral components'''
    N = len(data)
    half_N = round(N/2)
    
    # cut each trial in two:
    X = np.zeros([round(N/2), 2])
    X[:, 0 ] = data[:half_N]; X[:, 1] = data[half_N:]; 
    
    # Sqeeze the data in time by taking each second sample, odd and even
    X_2     = np.zeros([round(N/2), 2])
    X_2[:, 0] = data[range(0, N, 2)];  X_2[:, 1] = data[range(1, N, 2)]

    # Double each sample to eliminate the Hurst exponent, 1st ans 2nd halves
    double_X= np.zeros([round(N * 2)])
    double_X[range(0, N * 2, 2)] = data; double_X[range(1, N * 2, 2)] = data;
    X_1_2   = np.zeros([half_N, 2])
    X_1_2[:, 0] = double_X[:half_N]; X_1_2[:, 1] = double_X[half_N: N]; 

    windows = signal.windows.dpss(half_N, 2.5, Kmax = 5)
    ptime = 2; Sxx = []; Sxx_h_norm = []
    for t in range(2):
        for win in windows:
            Fx1 = np.fft.fft(  padding(X[:, t] * win,  ptime ) )
            Fxh = np.fft.fft(  padding(X_2[:, t] * win,  ptime ) )
            Fx1h = np.fft.fft(  padding(X_1_2[:, t] * win,  ptime ) )
            Sxx.append(Fx1 * np.conj(Fx1))
            Sxx_h = Fx1 * np.conj(Fxh)
            Sxx_1h = Fx1 * np.conj(Fx1h)
            Sxx_h_norm.append(   np.sqrt(  np.abs(Sxx_h) * np.abs(Sxx_1h)  )  )
    
    df = ( sampling_frequency / 2 ) / (half_N )
    freq = np.arange( 0, (half_N )) * df
    
    raw = (np.mean(Sxx, axis = 0) / (df * half_N * ptime / 2)) [:half_N]
    fractal   = (np.mean(Sxx_h_norm, axis = 0) / (df * half_N * ptime / 2)) [:half_N]
    harmonic = raw - fractal
    return raw, freq, harmonic, fractal

def cnoise(sampling_frequency, color, nsamp, p):
    ''' make some colored noise for tests '''
    q = 1
    freq = sampling_frequency * np.arange(0, (nsamp / 2 - 1)) / nsamp
    x = np.random.normal(0, p, size=[nsamp])
    X = np.fft.fft(x)
    H = 2 / (freq**(color/2) + q)
    z = np.zeros(nsamp - len(H))
    H = np.hstack([H, z])
    Y = X * H
    y = np.real(np.fft.ifft(Y) * 2. * np.pi)
    return y

# TODO: Detrend MEG 
# TODO: Take care of line frequency
# todo: tAKE INTO ACCOUNT THE SHIFT BETWEEN PUPIL AND BRAIN

fname = 'rest01_stage2_rest_raw.fif'
subjects = ['AA', 'AC', 'AL', 'AR', 'AW', 'BJB', 'CW', 'DJ', 'EC', 'FSM', 'JA',
            'JC', 'JP', 'JS', 'LS', 'NA', 'NM', 'MC', 'SL', 'SM', 'SF', 
            'TL', 'TK']

all_ex1 = np.zeros((23, 272, 5)); all_ex1[:] = np.nan
all_ex2 = np.zeros((23, 272, 5)); all_ex2[:] = np.nan
sensors = HLTP.get_ROI('', 'O')[1]
group_percentile = np.arange(0., 100.1, 20);

for sub_id, sub in enumerate(subjects):
    raw_data = mne.io.read_raw_fif(HLTP.MEG_pro_dir + '/' + sub + '/' + fname, preload=True)
    raw_data = raw_data.resample(600)
    data = raw_data.get_data()
    pupil_data = data[HLTP.pupil_chan, :]
    pupil_data =  clean_pupil(pupil_data)
    
    plt.plot(pupil_data)
    
    picks = mne.pick_types(raw_data.info, meg = True, ref_meg = False, exclude = 'bads')
    MEG_data = data[picks, :]
    fs = raw_data.info['sfreq']
    
    samples =np.arange(0, MEG_data.shape[-1], fs/2).astype('int')
    pupil = np.zeros(len(samples)); pupil[:] = np.nan
    pupil_delay = int(fs/4)
    for n, samp1 in enumerate(samples):
        samp1 = samp1 + pupil_delay
        samp2 = samp1 + int(fs/2)
        pdata_sample = pupil_data[samp1:samp2]
        num_nan_samp = sum(np.isnan(pdata_sample))# see if generally much samples are not due to blink
        if num_nan_samp < pupil_delay:
            pupil[n] = np.nanmedian(pdata_sample)
        else: pupil[n] = np.nan
   
    samples = samples[~np.isnan(pupil)]   
    pupil =   pupil[~np.isnan(pupil)]   
    
    # equal boundaries
    p_group1 = np.digitize(pupil, np.linspace(pupil.min(), pupil.max(), 5))
    # equal number of trials
    p_group2 = np.digitize(pupil, np.percentile(pupil, group_percentile))
    
    exponents = np.zeros(len(samples));
    mean_ex = []
    for chan in sensors:
        for n, samp1 in enumerate(samples):
            samp2 = samp1 + int(fs)
            raw, freq, harmonic, fractal = get_spectral_components(
                    (MEG_data[chan, samp1:samp2] - 
                     MEG_data[chan, samp1:samp2].mean()), fs, 5)
            exp, a = get_oof_param(fractal, freq, freq_range = [15, 100])
            exponents[n] = exp
        mean_ex = [exponents[p_group1 == grp].mean() for grp in range(1, 6)]
        all_ex1[sub_id, chan, :] = mean_ex
        mean_ex = [exponents[p_group2 == grp].mean() for grp in range(1, 6)]
        all_ex2[sub_id, chan, :] = mean_ex

# PLOTS 
for chan in sensors:
    plt.figure()
    mean_ex = all_ex1[:, chan, :]
    plt.errorbar(range(1, 6), np.nanmean(mean_ex, axis = 0), 
                     np.nanstd(mean_ex, axis = 0) / np.sqrt(23))

mean_ex = np.nanmean(all_ex2[:, :, :], axis = 1)
plt.errorbar(range(1, 6), np.nanmean(mean_ex, axis = 0), 
                 np.nanstd(mean_ex, axis = 0) / np.sqrt(23))

mean_ex = np.nanmean(all_ex1[:, :, :], axis = 1)
plt.errorbar(range(1, 6), np.nanmean(mean_ex, axis = 0), 
                 np.nanstd(mean_ex, axis = 0) / np.sqrt(23))


plt.figure()
mean_ex = np.nanmean(all_ex2, axis = 1)
plt.figure()

for sub_id, sub in enumerate(subjects):    
    plt.figure()
    mean_ex = all_ex1[sub_id, :, :]
    plt.errorbar(range(1, 6), np.nanmean(mean_ex, axis = 0), 
                 np.nanstd(mean_ex, axis = 0) / np.sqrt(len(sensors)))

for sub_id, sub in enumerate(subjects):    
    plt.figure()
    mean_ex = all_ex2[sub_id, 270:272, :]
    plt.errorbar(range(1, 6), np.nanmean(mean_ex, axis = 0), 
                 np.std(mean_ex, axis = 0) / np.sqrt(272))
    mean_ex = all_ex2[sub_id, 56:59, :]
    plt.errorbar(range(1, 6), np.nanmean(mean_ex, axis = 0), 
                 np.std(mean_ex, axis = 0) / np.sqrt(272))
       
plt.errorbar(range(1, 6), np.mean(mean_ex, axis = 0), 
             np.std(mean_ex, axis = 0) / np.sqrt(272))
         
    
    
plt.errorbar(range(1, 6), np.mean(mean_ex), np.std(mean_ex) / np.sqrt(272))
plt.plot(pupil - pupil.mean());plt.plot(exponents - exponents.mean());

plt.plot(freq[2:500], raw[2:500], label = 'raw')
plt.plot(freq[2:500], fractal[2:500], label = 'fractal')
exp, a = get_oof_param(fractal, freq, freq_range = [10, 100])
plt.plot(freq, a * (freq ** exp))
plt.xscale('log');plt.yscale('log');

plt.legend()

## test functions
data = cnoise(fs, 2.5, 1000, 1)
freqs, psd = signal.welch(data, fs = fs)

raw, freq, harmonic, fractal = get_spectral_components(data, fs, 5)
plt.plot(np.log(freq), np.log(raw), label = 'raw')
#plt.plot(np.log(freqs), np.log(psd), label = 'welch')
plt.plot(np.log(freq), np.log(fractal), label = 'oof')
plt.legend()

data = cnoise(sf, 3, 1000, 1)
raw, freq, harmonic, fractal = get_spectral_components(data, fs, 5)
get_oof_param(fractal, freq, freq_range = [10, 500])




