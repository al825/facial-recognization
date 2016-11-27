# Image pre-processing
from scipy import ndimage
import numpy as np


global SIZE
SIZE = 96

# Normalization transformation
def imanorm(image, new_min=0, new_max=255):
    return (image-min(image)) * (new_max-new_min)/new_max + new_min

# Histogram equalization
def histeq(image,nbr_bins=256):
    """  Histogram equalization of a grayscale image. """
    # get image histogram
    imhist,bins = np.histogram(image,nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    ima = np.interp(image,bins[:-1],cdf)

    return ima


# Image derivatives
def imaderiv(ima):
    ima = ima.reshape(SIZE, SIZE)
    imx = np.zeros(ima.shape)
    ndimage.filters.sobel(ima,1,imx)
    imy = np.zeros(ima.shape)
    ndimage.filters.sobel(ima,1,imy)
    magnitude = np.sqrt(imx**2+imy**2)
    return magnitude.flatten()
    
    
    


