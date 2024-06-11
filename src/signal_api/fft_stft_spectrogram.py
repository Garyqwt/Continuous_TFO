##############fft and stft#########
from scipy.signal import find_peaks, spectrogram, stft, istft, get_window
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

class STFT:

    def __init__(self, data: np.ndarray, data_f, data_t, fs: float, 
                 spectrum_interval: float, window_length: float):
        """
        A Spectrogram. Fourier co-efficients over time

        :param data: 2D data (time x fourier co-eff)
        :param fs: sampling rate of the original data
        :param spectrum_interval: time interval between each spectrum (in Seconds)
        """
        self.data = data
        self.data_f = data_f 
        self.data_t = data_t
        self.fs = fs
        self.fft_n = data.shape[1]
        self.num_data_point = data.shape[0]
        self.spectrum_interval = spectrum_interval
        self.window_length = window_length

    def freq_to_index(self, frequency):
        """
        Convert the given frequency to the lowest column index for this spectrogram

        :param frequency: in Hz
        :return: Column index number
        """
        return (2*np.asarray(frequency) * self.fft_n / self.fs).astype(int)

    def index_to_freq(self, index):
        """
        Convert the given index to its related frequency

        :param index: index
        :return: Frequency in Hz
        """
        return np.asarray(index) * self.fs / self.fft_n / 2
    
    def inverse(self):
        window_length_samples = int(self.window_length * self.fs)
        window_array = get_window('hann', window_length_samples)
        noverlap_samples = window_length_samples - self.spectrum_interval*self.fs
        return istft(self.data.T, fs=self.fs, window=window_array, nperseg=window_length_samples, 
              noverlap=noverlap_samples, nfft=window_length_samples, boundary=None)
    
    def plot(self, frequency_range, ax, vmin=-130, vmax=-60, 
             xylabels=True, mhr=None, fhr=None, title='', cmap='viridis'):
        """
        Plot the given spectrogram. Make sure the format is in (time x fourier coe-effs). This does not create a new plot

        :param frequency_range: Range of frequency to keep in the plot (in Hz)
        :return: matplotlib axis image
        """
        im = ax.pcolormesh(self.data_f,
                           self.data_t/60, 
                           10*np.log10((np.abs(self.data))**2), 
                           cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlim([frequency_range[0], frequency_range[1]])
        if xylabels:
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Time [min]')
        if mhr is not None:
            mhr = mhr[:len(fhr)]
            tt = np.linspace(0, len(mhr) / 60, len(mhr))
            ax.plot(mhr, tt, color='cyan', alpha=0.8, linestyle='dashed')
        if fhr is not None:
            tt = np.linspace(0, len(fhr) / 60, len(fhr))
            ax.plot(fhr, tt, color='red', alpha=0.8, linestyle='dotted')
        ax.set_title(title)
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, label='Power Spectrum [dB]')
        return im
#         plt.colorbar(im, ax=ax, label='Power Spectrum [dB]')

    
def generate_STFT(signal: np.ndarray, fs: float, window_length: float = 60, overlap_percent: float = 3 / 4,
                         window_type: str = 'hann'):
    """
    Generate a spectrogram from the given 1D Signal

    :param signal: 1D data
    :param fs: Sampling Frequency (Hz)
    :param window_length: Length(s) of each spectrogram window
    :param overlap_percent: 0 < p < 1, percentage overlap in spectrogram
    :param window_type: Type of window
    :return: custom Spectrogram object with plotting options
    """
    window_length_samples = int(window_length * fs)
    overlap_length_samples = int(window_length_samples * overlap_percent)
    spectrum_interval_length_samples = window_length_samples - overlap_length_samples
    window_array = get_window('hann', window_length_samples)
    
    f, t, Zxx = stft(signal, fs=fs, noverlap=overlap_length_samples, window=window_array, nperseg=window_length_samples, nfft=window_length_samples, boundary=None, padded=False)

    return STFT(data=Zxx.T, data_f = f, data_t = t, fs=fs, spectrum_interval=spectrum_interval_length_samples / fs, window_length= window_length)


def plt_fft(samples, samplerate, signal):
    N = samples
    # sample spacing
    T = 1.0 / samplerate
    x = np.linspace(0.0, N*T, N)

    yf = np.fft.fft(signal)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    plt.figure()
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.xlim((0,6))
    plt.xlabel("Freq (Hz)")
    plt.ylabel("Amplitude")

def plot_spectrogram(ppg_data, fhr, mhr, shp_rd, filepath='../runtime/good_fetal_spectrogram/', best_d=3, vmin=None, vmax=None):
    '''
    This is a function that took 5 detectors data from both WLs and plot two plots, each
    containing 6 spectrograms including the best detector spectrogram with FHR tracing 
    and all 5 detectors spectrograms in sequence.
    '''
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(30,10))
    axs = axs.flatten()
    generate_STFT(ppg_data[:,best_d], 80).plot((0,5),axs[0], fhr=fhr, mhr=mhr, vmin=vmin, vmax=vmax, title = f'D{best_d+1} 740nm')
    for i in range(5):
        generate_STFT(ppg_data[:,i], 80).plot((0,5),axs[i+1], vmin=vmin, vmax=vmax, title = f'D{i+1} 740nm')
    fig.suptitle(shp_rd+' (740nm)', fontsize=20, y=0.95)
    filename = filepath + 'spectrogram_740_' + shp_rd
    plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', bbox_inches='tight', pad_inches=0.1)

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(30,10))
    axs = axs.flatten()
    generate_STFT(ppg_data[:,best_d+5], 80).plot((0,5),axs[0], fhr=fhr, mhr=mhr, vmin=vmin, vmax=vmax, title = f'D{best_d+1} 850nm')
    for i in range(5):
        generate_STFT(ppg_data[:,i+5], 80).plot((0,5),axs[i+1], vmin=vmin, vmax=vmax, title = f'D{i+1} 850nm')
    fig.suptitle(shp_rd+' (850nm)', fontsize=20, y=0.95)
    filename = filepath + 'spectrogram_850_' + shp_rd
    plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', bbox_inches='tight', pad_inches=0.1)
    
# def my_spect_plot(sig_740, sig_850, name, detnum, fhr = [], mhr = [], Fs = 80):
#     #helper function to plot spectrogram of same detector with two wavelengths and add fhr and mhr highlights
#     f, axs = plt.subplots(1, 2, figsize=(15,4))
#     plot_spectrogram(sig_740, Fs, axs[0], '{} 740 det{}'.format(name, detnum), vmin=-200, vmax=-60)
#     plot_spectrogram(sig_850, Fs, axs[1], '{} 850 det{}'.format(name, detnum), vmin=-200, vmax=-60)
#     if len(fhr)>0:
#         axs[0].plot(np.repeat(fhr, Fs)+0.1, np.arange(Fs*len(fhr))/Fs/60, alpha=0.3, color='red')
#         axs[1].plot(np.repeat(fhr, Fs)+0.1, np.arange(Fs*len(fhr))/Fs/60, alpha=0.3, color='red')
#         axs[0].plot(np.repeat(fhr, Fs)-0.1, np.arange(Fs*len(fhr))/Fs/60, alpha=0.3, color='red')
#         axs[1].plot(np.repeat(fhr, Fs)-0.1, np.arange(Fs*len(fhr))/Fs/60, alpha=0.3, color='red')
#     if len(mhr)>0:
#         axs[0].plot(np.repeat(mhr, Fs)+0.1, np.arange(Fs*len(mhr))/Fs/60, alpha=0.3, color='blue', linewidth=3)
#         axs[1].plot(np.repeat(mhr, Fs)+0.1, np.arange(Fs*len(mhr))/Fs/60, alpha=0.3, color='blue', linewidth=3)    
#         axs[0].plot(np.repeat(mhr, Fs)-0.1, np.arange(Fs*len(mhr))/Fs/60, alpha=0.3, color='blue', linewidth=3)
#         axs[1].plot(np.repeat(mhr, Fs)-0.1, np.arange(Fs*len(mhr))/Fs/60, alpha=0.3, color='blue', linewidth=3)
#     return axs

def plot_spectrogram_row(ppg_data, shp_rd, filepath='../runtime/good_fetal_spectrogram/', vmin=None, vmax=None):
    '''
    This is a function that took 5 detectors data from both WLs and plot two plots, each
    containing 6 spectrograms including the best detector spectrogram with FHR tracing 
    and all 5 detectors spectrograms in sequence.
    '''
    fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(30,5))
    axs = axs.flatten()
    for i in range(5):
        generate_STFT(ppg_data[:,i], 80).plot((0,5),axs[i], vmin=vmin, vmax=vmax, title = f'D{i+1} 740nm')
    fig.suptitle(shp_rd+' (740nm)', fontsize=20, y=1.1)
    filename = filepath + 'spectrogram_row_740_' + shp_rd
    plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', bbox_inches='tight', pad_inches=0.1)

    fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(30,5))
    axs = axs.flatten()
    for i in range(5):
        generate_STFT(ppg_data[:,i+5], 80).plot((0,5),axs[i], vmin=vmin, vmax=vmax, title = f'D{i+1} 850nm')
    fig.suptitle(shp_rd+' (850nm)', fontsize=20, y=1.1)
    filename = filepath + 'spectrogram_row_850_' + shp_rd
    plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', bbox_inches='tight', pad_inches=0.1)
