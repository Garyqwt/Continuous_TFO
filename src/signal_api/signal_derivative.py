import numpy as np

def normalized_derivative(signal, step=1):
    """
    Calculates the normalized derivative of a time-series signal.

    Parameters:
        signal (list or numpy array): The input time-series signal.
        step (integer): The step for calculating derivatives in terms of numer of points.
            Default is 1 (i.e. x[n+1]-x[n]).
        # pad_value (bool): 
        #     The value to use for padding the first data point of the output.
        #     If False, no padding is added. 
        #     If True, 0 is used as the pad value.

    Returns:
        A numpy array containing the normalized derivative of the input signal, padded with pad_value if specified.
    """
    # Convert the signal to a numpy array.
    signal = np.array(signal)

    signal_shift = signal[step:]
    signal_shift = np.pad(signal_shift, (0, step), mode='edge')

    # Calculate the difference between every two consecutive data points.
    diff = signal - signal_shift

    # Calculate the average of every two consecutive data points.
    avg = np.mean([signal, signal_shift], axis=0)

    # Calculate the normalized derivative as the difference divided by the average.
    nd = diff/avg

    #  # Determine the pad value, if any.
    # if padding is True:
    #     pad_value = 0
    #     # Pad the first data point of the output with pad_value.
    #     nd = np.insert(nd, 0, pad_value)
    #     diff = np.insert(diff, 0, pad_value)

    return diff, nd