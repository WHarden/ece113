from venv import create
import numpy as np
import matplotlib.pyplot as plt 

def dft(signal : np.array) -> np.array:
    """Returns the Discrete Fourier Transform of a signal

    Args:
        signal (np.array): Input signal

    Returns:
        np.array: Input Signal in the Frequency Domain
    """

    #1. Generate the DFT matrix by completing the <create_dft_matrix> function.
    length_signal = len(signal)
    dft_matrix = create_dft_matrix(length=length_signal)

    #2. Apply the DFT matrix to the signal by completing the <dft_matrix_multiply> function.
    #   Store the frequency domain representation of the signal in variable <F_Signal>
    F_signal = apply_dft_matrix(dft_matrix=dft_matrix, signal=signal)

    return F_signal

def create_dft_matrix(length : int) -> np.array:
    """Returns a DFT matrix to enable calculation of the DFT.

    Args:
        length (int): length of the input signal or N of the NxN DFT Matrix.

    Returns:
        np.array: NxN DFT Matrix
    """
    dft_mat = np.zeros(shape=(length,length), dtype=np.cdouble)

    #***************************** Please add your code implementation under this line *****************************
    # Hint: The complex number e^(-4j) can be represented in numpy by <np.exp(-4j)>
    N = length
    indices = np.arange(0, N, dtype=np.int32)
    for i in range(N):
        dft_mat[i,:] = np.exp(-1j*2*np.pi*i*indices/N)
    #***************************** Please add your code implementation above this line *****************************

    return dft_mat

def apply_dft_matrix(dft_matrix : np.array, signal : np.array):
    """_summary_

    Args:
        dft_matrix (np.array): _description_
        signal (np.array): _description_

    Returns:
        _type_: _description_
    """
    signal_length = signal.shape[0]
    F_signal = np.zeros(shape=signal_length)

    #***************************** Please add your code implementation under this line *****************************
    # Hint: search the numpy library documentation for the <np.matmul> operation.
    F_signal = dft_matrix @ signal
    #***************************** Please add your code implementation above this line *****************************

    return F_signal

def plot_dft_magnitude_angle(frequency_axis : np.array, f_signal : np.array, fs=1, format=None):
    
    N = len(f_signal)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)

    #***************************** Please add your code implementation under this line *****************************
    # Hint: search the numpy library documentation for the <np.matmul> operation.
    #ax1.plot(..., ...)
    #ax2.plot(..., ...)
    if(format == None):
        ax1.set_ylabel("Magnitude")
        ax1.stem(frequency_axis, np.abs(f_signal))
        ax2.set_ylabel("Phase (radians)")
        ax2.stem(frequency_axis, np.angle(f_signal))
        ax2.set_xlabel("Frequency Bins")
        ax2.set_ylim((-np.pi, np.pi))
    if(format == "ZeroPhase"):
        #Hint: the numpy library documentation for the <np.where> operation might be useful.
        epsilon = 1e-4
        ax1.set_ylabel("Magnitude")
        ax1.stem(frequency_axis, np.abs(f_signal))
        ax2.set_ylabel("Phase (radians)")
        phase_arr = np.angle(f_signal)
        phase_arr[np.where(np.abs(f_signal) < epsilon)] = 0
        ax2.stem(frequency_axis, phase_arr)
        ax2.set_xlabel("Frequency Bins")
        ax2.set_ylim((-np.pi, np.pi))
    if(format == "Normalized"):
        frequency_axis = frequency_axis / (len(f_signal))
        epsilon = 1e-4
        ax1.set_ylabel("Magnitude")
        ax1.stem(frequency_axis, np.abs(f_signal))
        ax2.set_ylabel("Phase (radians)")
        phase_arr = np.angle(f_signal)
        phase_arr[np.where(np.abs(f_signal) < epsilon)] = 0
        ax2.stem(frequency_axis, phase_arr)
        ax2.set_xlabel("Frequency Bins")
        ax2.set_ylim((-np.pi, np.pi))
    if(format == "Centered_Normalized"):
        temp_f_signal = np.copy(f_signal)
        temp_f_signal[:int(N/2)] = f_signal[int(N/2):]
        temp_f_signal[int(N/2):] = f_signal[:int(N/2)]

        frequency_axis = frequency_axis / (len(f_signal)) - 0.5
        epsilon = 1e-4
        ax1.set_ylabel("Magnitude")
        ax1.stem(frequency_axis, np.abs(temp_f_signal))
        ax2.set_ylabel("Phase (radians)")
        phase_arr = np.angle(temp_f_signal)
        phase_arr[np.where(np.abs(temp_f_signal) < epsilon)] = 0
        ax2.stem(frequency_axis, phase_arr)
        ax2.set_xlabel("Frequency Bins")
        ax2.set_ylim((-np.pi, np.pi))
    if(format == "Centered_Original_Scale"):
        temp_f_signal = np.copy(f_signal)
        temp_f_signal[:int(N/2)] = f_signal[int(N/2):]
        temp_f_signal[int(N/2):] = f_signal[:int(N/2)]

        frequency_axis = frequency_axis / (len(f_signal)) - 0.5
        frequency_axis *= fs
        epsilon = 1e-4
        ax1.set_ylabel("Magnitude")
        ax1.stem(frequency_axis, np.abs(temp_f_signal))
        ax2.set_ylabel("Phase (radians)")
        phase_arr = np.angle(temp_f_signal)
        phase_arr[np.where(np.abs(temp_f_signal) < epsilon)] = 0
        ax2.stem(frequency_axis, phase_arr)
        ax2.set_xlabel("Frequency Bins")
        ax2.set_ylim((-np.pi, np.pi))
    #***************************** Please add your code implementation above this line *****************************

def idft(signal : np.array) -> np.array:
    """Returns the Inverse Discrete Fourier Transform of a frequency domain signal. 

    Args:
        signal (np.array): Input frequency signal

    Returns:
        np.array: Transforms input signal to time-domain.
    """
    length_signal = len(signal)
    time_domain_signal = np.zeros(length_signal)

    #***************************** Please add your code implementation under this line *****************************
    # Try to only use the <create_dft_matrix> and <apply_dft_matrix> operations that you have already implemented and some numpy operations.
    # Hint: look up the <np.conjugate>
    dft_matrix = create_dft_matrix(length=length_signal)
    dft_matrix = np.conjugate(dft_matrix)
    time_domain_signal = apply_dft_matrix(dft_matrix=dft_matrix, signal=signal) / length_signal
    #***************************** Please add your code implementation above this line *****************************

    return time_domain_signal

def convolve_signals(x_signal : np.array, y_signal : np.array) -> np.array:
    
    z_signal = np.zeros(len(x_signal))
    
    #***************************** Please add your code implementation under this line *****************************
    length = np.max((len(x_signal), len(y_signal)))
    F_x_signal = dft(x_signal)
    F_y_signal = dft(y_signal)
    F_z_signal = F_x_signal*F_y_signal
    z_signal = idft(F_z_signal)
    #***************************** Please add your code implementation above this line *****************************

    return z_signal


def zero_pad_signal(x_signal : np.array, new_length : int) -> np.array:
    
    zero_padded_signal = np.zeros(new_length)

    #***************************** Please add your code implementation under this line *****************************
    zero_padded_signal[0:len(x_signal)] = x_signal
    #***************************** Please add your code implementation above this line *****************************

    return zero_padded_signal
