import sys
import cv2
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import matplotlib.colors as clr

#parse commands
def main():
    model = 1
    filename = "images/moonlanding.png" #TODO Change this at the end to the root folder
    num_arg = len(sys.argv)
    print("number of arguments " + str(num_arg))
    
    match num_arg:
        case 5:
            print("5 arguments found")
            model = sys.argv[2]
            filename = sys.argv[4]
            # print(model)
            # print(filename)
        case 3:  
            print("3 arguments found")
            if (sys.argv[1] == "-m"):
                model = sys.argv[2]
            elif (sys.argv[1] == "-i"):
                filename = sys.argv[2]
        case _: 
            print("Error: invalid arguments found")
            pass
    da_int_model = int(model)
    match da_int_model:
        case 1:
            print("entering model 1")
            #performing ftt 2d on the array
            img_arr1 = image_convert(filename)
            fft_2d_img_1 = fft_2d(img_arr1)

            print(fft_2d_img_1)
            print(np.fft.fft2(img_arr1))

            #convert to float
            fft_2d_img_1 = np.real(fft_2d_img_1)

            #creating the graph
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(img_arr1, cmap='gray')
            axs[0].set_title('OG image') #TODO change the name lmao
            axs[1].imshow(np.abs(fft_2d_img_1), norm=clr.LogNorm(vmin=5), cmap='gray')
            axs[1].set_title('2D FFT LOG')
            plt.show()
        case 2:
            print("Entering mode 2")
            # Denoise the original image
            original = image_convert(filename)
            # FFT first
            fft_img = fft_2d(original.copy())

            # Set high frequencies to zero
            # im_fft2 = fft_img.copy()

            # keep_fraction = 0.01

            # # Set r and c to be the number of rows and columns of the array.
            # r, c = im_fft2.shape

            # # Set to zero all rows with indices between r*keep_fraction and
            # # r*(1-keep_fraction):
            # im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0

            # # Similarly with the columns:
            # im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

            # fft_2d_img_inversed = fft_2d_inverse(im_fft2).real

            # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            # ax[0].imshow(original,  # norm=LogNorm(),
            #          cmap='gray',
            #          interpolation='none')
            # ax[1].imshow(fft_2d_img_inversed,  # norm=LogNorm(),
            #          cmap='gray',
            #          interpolation='none')

            # plt.show()
        case 3:
            #perform fft 2d on array
            img_arr3 = image_convert(filename)
            fft_2d_img_3 = fft_2d(img_arr3)

            #convert to float
            fft_2d_img_3 = np.real(fft_2d_img_3)

            #creating the graph
            fig, axs = plt.subplots(2, 3, figsize=(10, 10))
            fig.suptitle('Compressed images at different levels')
            axs[0, 0].imshow(img_arr3, cmap='gray')
            axs[0, 0].set_title('Original Image')
            axs[0, 1].imshow(np.abs(fft_2d_img_3), norm=clr.LogNorm(vmin=5), cmap='gray')
            axs[0, 1].set_title('2D FFT LOG')

            c_percentage = [0, 25, 50, 65, 80, 95]
            for i, l in enumerate(c_percentage):
                c_fft = fft_2d_img_3.copy()
                # set a percentage of the coefficients to zero based on the compression level
                c_fft[int(c_fft.shape[0] * l / 100):, :] = 0
                c_fft[:, int(c_fft.shape[1] * l / 100):] = 0
                c_img = np.fft.ifft2(c_fft).real
                axs[i//3, i%3].imshow(c_img, cmap='gray')
                axs[i//3, i%3].set_title(str(l) + '% Compression')

            plt.show()
            # c_img = Image.fromarray(c_arr.astype(np.uint8))
            # c_img.save("outputimage/model_3_image.jpg")

        case _:
            pass


'''Optimized functions below'''

# Helper method to reduce copy pasted code for 2d functions
# Will take in the array and function to work with and return transformed numpy array
def run_2d(func, arr):
    # obtain rows and columns to loop over
    shape = arr.shape
    rows = shape[0]
    cols = shape[1]

    result = np.zeros(shape, dtype=np.complex_)

    # get intermediate matrix
    # for each row, perform a FFT on it
    for i in range(rows):
        inter = func(arr[i])
        result[i] = inter
    
    # perform a second FFT on transpose
    result = result.T
    for i in range(cols):
        inter = func(result[i])
        result[i] = inter

    # transpose again to get correct shape
    result = result.T 
    
    return result
#
#
#
#  FFT
#
#
#

#very cool turkey algo
def fft(da_list): 
    n = len(da_list)
    
    #base case (runs with set threshold of 8 in order to optimize method calls)
    if n <= 8:
        return dft(da_list)
    
    #creating two lists for both even and odd indices
    even = fft(da_list[::2])
    odd = fft(da_list[1::2])
    
    numerator = -2j*np.pi*np.arange(n)
    f = np.exp(numerator/ n)
    result = np.concatenate([even + f[:int(n/2)] * odd,
                        even + f[int(n/2):] * odd])
    result = np.round(result, 9)
    return result

def fft_2d(arr):
    # Call helper with fft method as input
    return run_2d(fft, arr)

# FFT inverse, uses level parameter to ensure final result isn't scaled down too much
def fft_inverse(arr, level=0):
    n = len(arr)
    if n <= 8:
        return dft_inverse(arr)
    
    even = fft_inverse(arr[::2], level+1)
    odd = fft_inverse(arr[1::2], level+1)
    
    numerator = 2j*np.pi*np.arange(n)
    f = np.exp(numerator / n)
    result = np.concatenate([even + f[:int(n/2)] * odd,
                           even + f[int(n/2):] * odd])
    
    result = np.round(result, 9)

    # Once the original level is reached, scale the final result by n
    if level == 0:
        result = result / n
    
    return result

def fft_2d_inverse(arr):
    return run_2d(fft_inverse, arr)
    

#
#
#
#  DFT
#
#
#
  
# takes in a 1D numpy array and performs a DFT on it
def dft(arr):
    n = arr.shape[0] # first item in tuple is the number
    result = np.empty(n, dtype=np.complex_)

    for i in range(n):
        # compute exponent for transform (-2pi * i * k which is represented with np.arange each time)
        e = np.exp((-2j * math.pi * i * np.arange(n)) / n)
        result[i] = np.dot(arr, e) # the summation is simply a dot product of the original array with the exponent calc. above

    # round to 9 decimal places similar to numpy method
    result = np.round(result, 9)    
    return result

# takes in a 2D numpy array and perform a DFT on it
def dft_2d(arr):
    return run_2d(dft, arr)

# inverse dft
def dft_inverse(arr):
    n = arr.shape[0] # first item in tuple is the number
    result = np.empty(n, dtype=np.complex_)

    for i in range(n):
        # compute exponent for transform (-2pi * i * k which is represented with np.arange each time)
        e = np.exp((2j * math.pi * i * np.arange(n)) / n)
        result[i] = np.dot(arr, e) # the summation is simply a dot product of the original array with the exponent calc. above

    result = result / n
    # round to 9 decimal places similar to numpy method
    result = np.round(result, 9)    
    return result

# Method to convert input image to numpy matrix using cv2
def image_convert(image_name):
    img = cv2.imread(image_name, 0)
    # resize the image array to one that has size near a power of 2
    n = 2**int(np.ceil(np.log2(max(img.shape))))

    arr = cv2.resize(img, (n, n))
    return arr


if __name__ == '__main__':
    main()

    # original = image_convert("images/moonlanding.png")
    # fft_2d(original)

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # ax[0].imshow(original,  # norm=LogNorm(),
    #     cmap='gray',
    #     interpolation='none')
    
    # plt.show()


    # ABIOLA'S TESTS
    # arr = np.random.rand(2**5)
    # print(np.fft.ifft(arr))
    # print(dft_inverse(arr))
    # # print(fft_2d(arr))
    # print(dft_2d(arr))
    # print(fft_2d(arr))


    # ks = np.array(np.arange(3))

    # n = np.array(np.arange(3))
    # kn = ks.T * n
    # first_exponens = np.exp(-1j * 2 * math.pi * kn / 3)
    # a = np.matmul(np.asarray([1, 2, 3]), first_exponens.T)
    # print(a)