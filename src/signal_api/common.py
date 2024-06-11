import numpy as np
from scipy.signal import argrelextrema

def moving_average(signal, window_size):
    """
    Calculate the moving average of a given signal over a specified window size.

    Args:
    signal (array_like): Input array or object that can be converted to an array.
    window_size (int): The size of the window to take the average over.

    Returns:
    array: A smoothed array with the same shape as `signal`.
    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(signal, window, 'same')


def lower_envelope(signal, win_size=10):
    """
    Calculate the lower envelope of a given signal.

    Args:
    signal (array_like): Input array or object that can be converted to an array.

    Returns:
    array: An array representing the lower envelope of the input signal.
    """
    # Find the local minima indices
    minima_indices = argrelextrema(signal, np.less)[0]

    # Extract the values at the minima indices
    envelope_values = signal[minima_indices]

    # Create a piecewise linear function using the minima indices and values
    envelope_function = np.poly1d(np.polyfit(minima_indices, envelope_values, deg=1))

    # Evaluate the function over the entire range of the signal
    envelope = envelope_function(np.arange(len(signal)))

    return envelope