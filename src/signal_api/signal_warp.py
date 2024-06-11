import numpy as np
from scipy.interpolate import interp1d
from .fft_stft_spectrogram import *
from .filters import *

def interpolate_by_x(x0, y0, x1):
    '''
    given the (x0,y0) coordinates, finds vertical 
    coordinates y1 at horizontal coordinates x1
    '''
    y_interp = interp1d(x0, y0, 
                        fill_value="extrapolate")
    y1 = y_interp(x1)
    return y1

def resample_by_phase(sig, freq_seq, Fs0, Fs_ph):
    '''
    resamples the constant-spaced samples in time to get 
    constant-spaced samples in phase
    
    This is equivalent of getting a signal with constant 
    frequency of 1 and sampling frequency of Fs_ph 
    
    **inputs
    sig: input signal (constant-spaced samples in time)
    freq_seq: the input signal's dynamic frequency with 
              sampling rate of Fs0
    Fs0: input sampling frequency per second
    Fs_ph: sampling frequncy per period (full phase rotation)
    
    **returns
    time2: time-stamp of resampled data
    sig2: resampled data
    '''
    phase0 = 2*np.math.pi*np.cumsum(freq_seq)/Fs0
    time0 = np.arange(len(phase0))/Fs0
    
    ph_increments = 2*np.math.pi/Fs_ph
    ph_selected = np.arange(0, 2*np.math.pi*np.sum(freq_seq)/Fs0, ph_increments)
    
    time1 = interpolate_by_x(phase0, time0, ph_selected)
    
    sig1 = interpolate_by_x(np.arange(len(sig))/Fs0,
                            sig, 
                            time1)
    return time1, sig1

def resample_filter(sig, Fs, fhr, bp_range = 0.1, plot=False, filename=None):
    t_uw, sig_uw = resample_by_phase(sig, np.repeat(fhr, Fs)[:len(sig)], Fs, Fs)
    sig_fil = butter_bandpass_filter(sig_uw, 1-bp_range, 1+bp_range, Fs, order=5)
    sig_fil_rw = interpolate_by_x(t_uw, sig_fil, np.arange(len(sig))/Fs)

    if plot:
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20,10))
        axs = axs.flatten()
        generate_STFT(sig, Fs).plot((0,5),axs[0], fhr=fhr, vmin=None, vmax=None, title = 'Original sig')
        generate_STFT(sig_uw, Fs).plot((0,5),axs[1], vmin=None, vmax=None, title = 'Unwarped around fhr')
        generate_STFT(sig_fil, Fs).plot((0,5),axs[2], vmin=None, vmax=None, title = 'Filter around 1Hz')
        generate_STFT(sig_fil_rw, Fs).plot((0,5),axs[3], vmin=None, vmax=None, title = 'Rewarped filtered fhr')
        plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', bbox_inches='tight', pad_inches=0.1)

    return sig_fil_rw