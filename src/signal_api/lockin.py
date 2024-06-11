from scipy.signal import butter, sosfiltfilt
import numpy as np
from .filters import butter_lowpass_filter

def myfilter_sos(inp, Wn, N, filter_type, plot = False):
    #filter_type = 'bandpass', 'high', 'low'
    
    sos = butter(N, Wn, filter_type, fs = 80, output = 'sos')
    
    if len(inp.shape) > 1:
        out_sig = np.empty_like(inp)
        for ch in range(inp.shape[1]):
            out_sig[:,ch] = sosfiltfilt(sos, inp[:,ch])
        return out_sig
    return sosfiltfilt(sos, inp)

def lockin_separation(HR, sig_TFO, Fs, cut_off_freq = 0.2):
    I = 2*np.sin(np.cumsum(HR)/Fs*2*np.math.pi)
    Q = 2*np.cos(np.cumsum(HR)/Fs*2*np.math.pi)
    
    I = np.multiply(sig_TFO[:len(HR)],I[:len(sig_TFO)])
    Q = np.multiply(sig_TFO[:len(HR)],Q[:len(sig_TFO)])
    order = 5  # filter order
    I = myfilter_sos(I, cut_off_freq, order, 'low')
    Q = myfilter_sos(Q, cut_off_freq, order, 'low')
    return np.sqrt(np.multiply(I,I)+np.multiply(Q,Q))  