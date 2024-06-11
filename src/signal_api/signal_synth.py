import numpy as np
from scipy.interpolate import interp1d

def sawtooth(t, bias = 0.5):
    '''
    sample function
    '''
    return t + bias

def sin(t, bias = 0.5):
    '''
    sample function
    '''
    return np.sin(2*np.math.pi*t) + bias

def cos(t, bias = 0.5):
    '''
    sample function
    '''
    return np.cos(2*np.math.pi*t) + bias


def generate_periodic_sig(t, 
                          fun = np.sin, 
                          period = 2*np.math.pi):
    '''
    t: input time stamps assuming signal period = period
    fun: input function with acceptable input from [0,1)
    period: the assumed period of input t
    
    on the input function if fun(0)!=fun(1-epsilon) 
    ''' 
    return fun((t%period)/period)

def synthesized_freq_sequence_repeat(freq_seq, Fs, amp_seq = [], apply_bias = False):
    '''
    repeats freq_seq elements for a full period based on their frequency values
    freq_seq: raw frequency sequence of the main harmonic
    Fs: output sampling freq
    
    return
    the repeated frequency sequence
    '''
    repeat = np.int32(np.round((1/freq_seq)*Fs))
    #bias changes the input frequencies slightly to get the full period after repeats
    if apply_bias:
        bias = np.round((Fs - freq_seq*repeat - 1e-10)/repeat,5)
        Fr = np.concatenate(([0], np.repeat(freq_seq+bias, repeat)),
                            axis = 0)
    else:
        Fr = np.concatenate(([0], np.repeat(freq_seq, repeat)),
                            axis = 0)

    if len(amp_seq) == 0:
        amp = np.ones_like(Fr)
    else:
        amp = np.concatenate(([amp_seq[0]], np.repeat(amp_seq, repeat)),
                             axis = 0)
    return Fr, amp

def generate_sigfromfunc(freq_seq_raw, Fs, fun, amp_seq_raw = [], apply_bias = False):
    '''
    freq_seq: raw frequency sequence of the main harmonic
    Fs: output sampling freq
    fun: a function fun(x) that generates a full period of 
         signal by receiving the input x values from [0,1)
    amp_seq_raw: raw amp sequence of the signal
    
    returns
    t: time stamps with frequency=Fs
    sig: the generated signal
    Fr: signal's main harmonic frequency for each time stamp t
    
    example usage:
    generate_sigfromfunc(freqs, Fs, sin)
    generate_sigfromfunc(freqs, Fs, sawtooth)
    '''
    Fr, Amp = synthesized_freq_sequence_repeat(freq_seq_raw, 
                                               Fs, 
                                               amp_seq_raw, 
                                               apply_bias = apply_bias)
    # Fr = freq_seq_raw
    # Amp = amp_seq_raw

    t = np.arange(len(Fr))/Fs

    sig = generate_periodic_sig(np.cumsum(Fr)/Fs*2*np.math.pi, 
                                fun = fun, 
                                period = 2*np.math.pi)
    return t, sig*Amp, Fr, Amp

def change_sampling_rate(sig, current_rate, target_rate):
    '''
    changes the sampling rate of the input signal
    sig: input time-serie
    current_rate: the input sampling rate
    target_rate: the target sampling rate
    
    returns
    target_signal: the resampled time serie
    '''
    
    rate_ratio = current_rate / target_rate
    current_time_indices = np.arange(len(sig))
    target_time_indices = np.arange(0, len(sig), rate_ratio)
    
    # Create an interpolation function based on the current and target time indices
    interpolator = interp1d(current_time_indices, sig, 
                            fill_value="extrapolate")
    target_signal = interpolator(target_time_indices)
    
    return target_signal