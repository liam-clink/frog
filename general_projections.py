from operator import index
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize
from numpy.random import default_rng
rng = default_rng()

folder = 'Second Stage/'

trace = np.loadtxt(folder+'processed_data.tsv', delimiter='\t')
trace /= np.max(trace)
delays = np.loadtxt(folder+'processed_data_delays.tsv', delimiter='\t')
frequencies = np.loadtxt(folder+'processed_data_freqs.tsv', delimiter='\t')

plt.imshow(trace)
plt.show()

# Spectrum of the original pulse (not necessary, but helpful)
original_spectrum = np.loadtxt(folder+'processed_spectrum.tsv')
original_spectrum[:,1] /= np.max(original_spectrum[:,1])
original_frequencies = original_spectrum[:,0]
original_spectrum[:,1] -= 0.0075
# Change negative values to zero
index_filter = original_spectrum[:,1] < 0.
original_spectrum[index_filter,1] = 0.
original_spectral_amplitude = original_spectrum[:,1]**0.5

# Calculate pulse from spectrum
# FFT produces a spectrum centered at zero, IFFT expects the same
# This means that FFT calculates spectrum of the envelope, unless the spectrum
# is padded so that the negative range of frequencies equals the positive range
timestep = 1./(original_frequencies[-1]-original_frequencies[0])
grid_size = len(original_frequencies) #TODO: throw error if odd grid size, or grids don't match
duration = grid_size*timestep 
times = np.fft.ifftshift(np.fft.fftfreq(grid_size, 1./duration))
shifted_original_frequencies = np.fft.ifftshift(np.fft.fftfreq(grid_size, timestep))
initial_scale = 8. # Change to make initial SHG have similar scale to trace
initial_guess = initial_scale*np.fft.ifftshift(np.fft.ifft(original_spectral_amplitude))


fig, axes = plt.subplots(1, 2)
axes[0].plot(shifted_original_frequencies, original_spectral_amplitude)
axes[0].set_title('Original Spectrum')
axes[1].plot(times, initial_guess.real)
axes[1].plot(times, initial_guess.imag)
#axes[1].plot(times, abs(new_guess*np.conj(new_guess)))
#axes_phase = axes[1].twinx()
#axes_phase.scatter(times, np.angle(new_guess), color='red')
axes[1].set_title('Flat Spectral Phase Pulse')
plt.show()


# Define functions for error calculation
def calc_trace(pulse, delays):
    pulse_interpolator = scipy.interpolate.interp1d(times, pulse, bounds_error=False, fill_value=0.)
    trace = np.zeros((len(delays), len(pulse)))
    for i in range(len(delays)):
        gate_pulse = pulse_interpolator(times-delays[i])
        shg = pulse*gate_pulse
        shg_fft = np.fft.fftshift(np.fft.fft(shg))
        trace[i,:] = abs(shg_fft*np.conj(shg_fft))
    return trace

def scale_factor(trace_measured, trace_calculated):
    return sum(trace_measured*trace_calculated)/sum(trace_calculated*trace_calculated)

# Calculate norm for error function, since it need only be calculated once
norm = trace.shape[0]*trace.shape[1]*np.max(trace)**2
def calc_error(pulse, trace_measured, delays):
    # Calculate trace
    trace_calculated = calc_trace(pulse, delays)
    # Calculate scale factor
    mu = scale_factor(trace_measured, trace_calculated)
    # Sum squares of residuals
    r = np.sum((trace_measured-mu*trace_calculated)**2)
    # Return normalized error
    return np.sqrt(r/norm)

# I'm using frequency instead of angular frequency because that is
# numpy's convention, and it's thus simpler this way

# FFT produces a spectrum centered at zero, IFFT expects the same
#TODO: add error if shape is odd (assert ValueError)
#TODO: perhaps shifting not being perfect is the issue...
shifted_frequencies = frequencies - frequencies[int(frequencies.shape[0]/2)]

# Uncomment to assess soft threshold level visually
'''
for i in range(trace.shape[-1]):
    plt.plot(shifted_frequencies, trace[i,:])
plt.show()
'''

def Z(E_new, E_sig):
    N = len(E_new)
    running_sum = 0.
    for i in range(N):
        for j in range(N):
            residual = E_sig[i,j] - E_new[j]*np.roll(E_new,-i)[j] #TODO: is this roll correct?
            running_sum += residual*np.conj(residual)
    return abs(running_sum)

def grad_Z(E_old, E_sig):
    N = len(E_old)
    running_sum = 0.
    for i in range(N):
        shift_right = np.roll(E_old,i)
        shift_left = np.roll(E_old,-i)
        running_sum += -np.conj(E_sig[i,:])*shift_right + np.conj(E_old)*shift_right*np.conj(shift_right) \
                       +np.conj(np.roll(E_sig,-i,axis=1)[i,:])*shift_left + np.conj(E_old)*shift_left*np.conj(shift_left)
    running_sum = 2.*running_sum.real
    return running_sum + 1.j*running_sum

threshold = 1.5e-3    
iterations = 30
pulse = initial_guess
frog_errors = []
for i in range(iterations):
    #fig, axes = plt.subplots(1, 2)
    #axes[0].plot(times, pulse.real)
    #axes[0].plot(times, pulse.imag)

    shg = np.zeros((len(delays), len(times)), dtype=np.complex128)

    # Iterate through lines
    for j in range(len(delays)):
        # Calculate SHG
        pulse_interpolator = scipy.interpolate.interp1d(times, pulse, bounds_error=False, fill_value=0.)
        gate_pulse = pulse_interpolator(times-delays[j])
        shg[j,:] = pulse*gate_pulse
        
        # FFT(SHG)
        shg_fft = np.fft.fftshift(np.fft.fft(shg[j,:]))
        #axes[0].plot(shifted_frequencies, trace[j,:])
        #axes[1].plot(shifted_frequencies, shg_fft*np.conj(shg_fft))
        
        # Update the part above threshold
        index_filter = trace[j,:] >= threshold
        shg_fft[index_filter] = abs(trace[j,index_filter])**0.5*np.exp(1.0j*np.angle(shg_fft[index_filter]))
        #axes[0].plot(shifted_frequencies[index_filter], trace[j,index_filter])
        #axes[1].plot(shifted_frequencies[index_filter], abs(shg_fft[index_filter])**2)
        
        # IFFT(SHG)
        shg_new = np.fft.ifft(np.fft.ifftshift(shg_fft))
        #axes[0].plot(times, abs(shg)**2)
        #axes[1].plot(times, abs(shg_new)**2)
        
        shg[j,:] = shg_new

    ## Update E
    # First calculate Z gradient
    flattened_pulse = np.concatenate((pulse.real, 1.j*pulse.imag))
    scipy.optimize.line_search(Z, grad_Z, pulse, -grad_Z(pulse, shg), args=(shg,)) #TODO: CHECK THIS, MAY NOT WORK AS EXPECTED
    input()

    #plt.imshow(calc_trace(pulse, delays))
    #plt.show()    

    # Print frog error
    frog_error = calc_error(pulse, trace, delays)
    print("Iteration: ", i, "FROG Error: ", frog_error)
    frog_errors.append(frog_error)

    #axes[1].plot(times, pulse.real)
    #axes[1].plot(times, pulse.imag)
    #plt.show()

np.savetxt(folder+'frog_errors.tsv', np.array(frog_errors), delimiter='\t')

#plt.plot(times, pulse.real)
#plt.plot(times, pulse.imag)
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(shifted_frequencies, delays, calc_trace(pulse, delays))#, shading='flat')
ax.set_title('Trace of Retrieved Pulse')
ax.set_xlabel('frequencies (Hz)')
ax.set_ylabel('delays (s)')
plt.savefig(folder+'final_trace.png', dpi=600)
plt.show()

# Final Result Comparison
fig, axes = plt.subplots(2, 2)

axes[0,0].plot(times, abs(pulse)**2)
axes[0,0].set_xlabel('time (s)')
axes[0,0].set_ylabel('amplitude (a.u.)')
index_filter = abs(pulse)**2 > np.max(abs(pulse)**2)/3.
ax_phase = axes[0,0].twinx()
ax_phase.plot(times[index_filter], np.unwrap(np.angle(pulse[index_filter])), color='red')
ax_phase.set_ylabel('phase (rad)')
axes[0,0].set_title('Retrieved Pulse')

original_spectrum = abs(original_spectral_amplitude)**2
axes[0,1].plot(shifted_frequencies, original_spectrum/np.max(original_spectrum))
retrieved_spectral_amplitude = np.fft.fftshift(np.fft.fft(pulse))
retrieved_spectrum = abs(retrieved_spectral_amplitude)**2
axes[0,1].plot(shifted_frequencies, retrieved_spectrum/np.max(retrieved_spectrum))
axes[0,1].set_xlabel('frequencies (Hz)')
index_filter = retrieved_spectrum > np.max(retrieved_spectrum)/10.
ax_phase = axes[0,1].twinx()
ax_phase.plot(shifted_frequencies[index_filter], np.unwrap(np.angle(retrieved_spectral_amplitude[index_filter])), color='red')
axes[0,1].set_title('Spectrum of Retrieved Pulse')

axes[1,0].pcolormesh(shifted_frequencies, delays, trace)
axes[1,0].set_title('Trace of Initial Guess Pulse')
axes[1,0].set_xlabel('frequencies (Hz)')
axes[1,0].set_ylabel('delays (s)')

axes[1,1].pcolormesh(shifted_frequencies, delays, calc_trace(pulse, delays))
axes[1,1].set_title('Trace of Retrieved Pulse')
axes[1,1].set_xlabel('frequencies (Hz)')
axes[1,1].set_ylabel('delays (s)')

plt.tight_layout()
plt.savefig(folder+'final_pulse.png', dpi=600)
plt.show()


# Calculate Dispersion values
peak_index = np.argmax(abs(pulse*np.conj(pulse)))
phase = np.unwrap(np.angle(pulse))
center_frequency = (phase[peak_index+1] - phase[peak_index-1])/(times[peak_index+1] - times[peak_index-1])
center_frequency /= 2.*np.pi
print(center_frequency)
phase = np.unwrap(np.angle(retrieved_spectral_amplitude))
group_delay_dispersion = (phase[peak_index+1] - 2*phase[peak_index] + phase[peak_index-1])/(times[peak_index+1] - times[peak_index-1])**2


#TODO: add error if shifted_frequencies != shifted_original frequencies
#plt.pcolormesh(shifted_frequencies, delays, trace)
#plt.show()