import numpy as np
from PIL import Image, ImageOps
import cv2 as cv #opencv
import matplotlib.pyplot as plt

folder = './First Stage/'
filenames = ['2um_Merge_9.tif']

'''
img_list = [Image.open(folder+fn) for fn in filenames]
img_list = [ImageOps.grayscale(img) for img in img_list]
data_list = [np.array(img) for img in img_list]
print(data_list[0].dtype)

# Merge exposures to HDR image
exposure_times = np.array([2.0, 4.0], dtype=np.float32) # exposure in s
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(data_list, times=exposure_times.copy())
#merge_robertson = cv.createMergeRobertson()
#hdr_robertson = merge_robertson.process(data_list, times=exposure_times.copy())
'''
'''
# Tonemap HDR image
tonemap1 = cv.createTonemap(gamma=2.2)
# Must convert to 3 channel for tonemapping to work...
# https://docs.opencv.org/3.4/d8/d5e/classcv_1_1Tonemap.html
# src	source image - CV_32FC3 Mat (float 32 bits 3 channels) 
# https://stackoverflow.com/questions/12191211/difference-between-cv-32fc3-and-cv-64fc3-in-opencv
# https://stackoverflow.com/questions/32171917/copy-2d-array-into-3rd-dimension-n-times-python
three_channel = np.repeat(hdr_debevec[:, :, np.newaxis], 3, axis=2)
res_debevec = tonemap1.process(three_channel)
print((res_debevec<0).sum())

processed_image = Image.fromarray(res_debevec)
processed_image.show()
'''

# Read the 16 bit tif image, which is just the 12 bit output with padding
raw_data = cv.imread(folder+filenames[0], cv.IMREAD_UNCHANGED)
# Returns a numpy array with dtype uint16

# Shift minimum to zero
shifted_data = raw_data - np.min(raw_data)
plt.imshow(raw_data)
plt.show()
# First index is vertical axis going downwards, second index is horizontal axis
## MAKE SURE THAT PEAK IS CENTERED VERTICALLY
cropped_data = shifted_data[110:1300,800:1200]
left_wavelength = -0.724*(900-1041) + 517 # nm per pixel
#plt.contour(cropped_data,levels=100)
#plt.show()


# Padding the spectrum to power of 2 for better FFT results
def is_pow_2(n):
    return (n & (n-1) == 0) and (n != 0)

if not is_pow_2(cropped_data.shape[1]):
    exponent = (int(np.ceil(np.log2(cropped_data.shape[1]))))
    new_width = 2**exponent
else:
    new_width = cropped_data.shape[1]

new_height = cropped_data.shape[0]

pad_left = int(np.floor((new_width-cropped_data.shape[1])/2.))
pad_right = new_width-cropped_data.shape[1]-pad_left

padded_data = np.pad(cropped_data, ((0,0),(pad_left,pad_right)))
new_left_wavelength = left_wavelength -0.724*(-pad_left)
right_wavelength = new_left_wavelength -0.724*padded_data.shape[1]
wavelengths = np.linspace(new_left_wavelength, right_wavelength, padded_data.shape[1],endpoint=True)
np.savetxt(folder+'image_wavelengths.tsv',wavelengths*1.e-9,delimiter='\t')

plt.contour(padded_data, levels=100)
plt.show()


print('output datatype: ', padded_data.dtype)
np.savetxt(folder+'processed_data.tsv', padded_data, delimiter='\t')
