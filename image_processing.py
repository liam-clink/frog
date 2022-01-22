import numpy as np
import cv2 as cv #opencv
import matplotlib.pyplot as plt
import scipy

folder = './Second Stage/'
filenames = ['final.png']

# Read the 16 bit hdr image
raw_data = cv.imread(folder+filenames[0], cv.IMREAD_UNCHANGED)
# Returns a numpy array with dtype uint16
# First index is vertical axis going downwards, second index is horizontal axis

green_calibration_wavelength = 517.e-9
green_calibration_pixel = 1041
wavelength_per_pixel = -0.724e-9 # m/px
delay_per_pixel = 0.1e-15 # s/px
crop_left_pixel = 356

delays = delay_per_pixel*np.linspace(-raw_data.shape[0]/2., raw_data.shape[0]/2., raw_data.shape[0])
wavelengths = np.arange(green_calibration_wavelength + wavelength_per_pixel*(crop_left_pixel-green_calibration_pixel),
                        green_calibration_wavelength + wavelength_per_pixel*raw_data.shape[1],
                        wavelength_per_pixel)
print(wavelengths[0], wavelengths[-1])

# Shift minimum to zero
shifted_data = raw_data - np.min(raw_data)
plt.imshow(raw_data)
plt.show()

left_wavelength = -0.724*(900-1041) + 517 # nm per pixel
new_left_wavelength = left_wavelength -0.724*356
print(left_wavelength, new_left_wavelength)
#plt.contour(cropped_data,levels=100)
#plt.show()


# Padding the spectrum to power of 2 for better FFT results
def is_pow_2(n):
    return (n & (n-1) == 0) and (n != 0)

if not is_pow_2(shifted_data.shape[1]):
    exponent = (int(np.ceil(np.log2(shifted_data.shape[1]))))
    new_width = 2**exponent

    scipy.interpolate.interp2d()
else:
    new_width = shifted_data.shape[1]


new_left_wavelength = left_wavelength -0.724*(-pad_left)
right_wavelength = new_left_wavelength -0.724*shifted_data.shape[1]
wavelengths = np.linspace(new_left_wavelength, right_wavelength, shifted_data.shape[1],endpoint=True)
np.savetxt(folder+'image_wavelengths.tsv',wavelengths*1.e-9,delimiter='\t')

plt.contour(shifted_data, levels=100)
plt.show()
print(wavelengths[0], wavelengths[-1])

print('output datatype: ', shifted_data.dtype)
np.savetxt(folder+'processed_data.tsv', shifted_data, delimiter='\t')
