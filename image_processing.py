import numpy as np
import cv2 as cv #opencv
import matplotlib.pyplot as plt
import scipy.interpolate

folder = './Second Stage/'
filenames = ['final.png']

# Read the 16 bit hdr image
raw_data = cv.imread(folder+filenames[0], cv.IMREAD_UNCHANGED)
# Returns a numpy array with dtype uint16
# First index is vertical axis going downwards, second index is horizontal axis

calibration_wavelength = 609.e-9 # m
calibration_pixel = 2029 # px
wavelength_per_pixel = 0.0903e-9 # m/px
delay_per_pixel = 6.e-15 # s/px
crop_left_pixel = 356

delays = delay_per_pixel*np.linspace(-raw_data.shape[0]/2., raw_data.shape[0]/2., raw_data.shape[0])
left_wavelength = calibration_wavelength + wavelength_per_pixel*(crop_left_pixel-calibration_pixel)
right_wavelength = left_wavelength + wavelength_per_pixel*raw_data.shape[1]
wavelengths = np.arange(left_wavelength, right_wavelength, wavelength_per_pixel)


# Shift minimum to zero
shifted_data = raw_data - np.min(raw_data)
plt.imshow(raw_data)
plt.show()


# FROG image should be square with power of 2 side length if using General Projection.
# Here the image is interpolated in case this isn't already the case.
# Also, the dt and dw need to match, so dw = 2*pi/(duration) and dt = 2*pi/(ang. freq. bandwidth)
old_size = max(shifted_data.shape[0], shifted_data.shape[1])
exponent = (int(np.ceil(np.log2(shifted_data.shape[1]))))
new_width = 2**exponent

interpolant = scipy.interpolate.interp2d(wavelengths, delays, shifted_data)
frequencies = np.linspace(2.99e8/right_wavelength, 2.99e8/left_wavelength, new_width, endpoint=True)
new_wavelengths = 2.99e8/frequencies
new_delays = np.linspace(delays[0], delays[-1], new_width, endpoint=True)
interpolated_data = interpolant(new_wavelengths, new_delays)

duration = new_delays[-1]-new_delays[0]
dt = duration/new_width
bandwidth = 2*np.pi*2.99e8*(1./wavelengths[0]-1./wavelengths[-1])
dw = bandwidth/new_width

print('actual duration: ', duration)
print('actual dt: ', dt)
print('dt from bandwidth: ', 2.*np.pi/bandwidth)
print('duration from dw: ', 2.*np.pi/dw)

print('actual bandwidth: ', bandwidth*1e-12)
print('actual dw: ', dw*1e-12)
print('dw from duration: ', 2.*np.pi/duration*1e-12)
print('bandwidth from dt: ', 2.*np.pi/dt*1e-12)

print('tbp: ', dt*dw, 'condition: ', 2.*np.pi/new_width)
#########

interpolant = scipy.interpolate.interp2d(wavelengths, delays, shifted_data)
angular_frequencies = np.linspace(2.*np.pi*2.99e8/right_wavelength, 2.*np.pi*2.99e8/left_wavelength, new_width, endpoint=True)
new_wavelengths = 2.*np.pi*2.99e8/angular_frequencies
new_delays = np.linspace(-new_width/2*(2.*np.pi/new_width/dw), new_width/2*(2.*np.pi/new_width/dw), new_width, endpoint=True)
interpolated_data = interpolant(new_wavelengths, new_delays)


np.savetxt(folder+'image_wavelengths.tsv', new_wavelengths, delimiter='\t')

plt.contour(angular_frequencies*1e-12, new_delays*1e15, interpolated_data, levels=100)
plt.xlabel('angular frequency (Trad/s)')
plt.ylabel('delay (fs)')
plt.title('Contour Plot of Interpolated Data')
print(interpolated_data.shape)
plt.show()

print('output datatype: ', interpolated_data.dtype)
np.savetxt(folder+'processed_data.tsv', interpolated_data, delimiter='\t')
np.savetxt(folder+'processed_data_delays.tsv', new_delays, delimiter='\t')
np.savetxt(folder+'processed_data_freqs.tsv', frequencies, delimiter='\t')
np.savetxt(folder+'processed_data_wavelengths.tsv', new_wavelengths, delimiter='\t')
