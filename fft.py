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
            print("no arguments found")
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
    #base case
    if n == 1:
        return dft(da_list)
    
    #creating two lists for both even and odd indices
    even = fft(da_list[::2])
    odd = fft(da_list[1::2])
    
    numerator = -2j*np.pi*np.arange(n)
    f = np.exp(numerator/ n)
    result = np.concatenate([even + f[:int(n/2)] * odd,
                           even + f[int(n/2):] * odd])
    return result

def fft_2d(arr):
    print("Entering fft_2d")
    # obtain rows and columns to loop over
    shape = arr.shape
    rows = shape[0]
    cols = shape[1]

    result = np.empty(shape, dtype=np.complex_)

    # get intermediate matrix
    # for each row, perform a FFT on it
    for i in range(rows):
        inter = fft(arr[i])
        result[i] = inter
    
    # perform a second FFT on transpose
    result = result.T
    for i in range(cols):
        inter = fft(result[i])
        result[i] = inter

    # transpose again to get correct shape
    result = result.T 
    
    return result

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
        dft_sum = 0 # reset the sum at each iteration of the loop
        for k in range(n):
            xn = arr[k]
            exp = np.exp((-2j * math.pi * i * k) / n)
            dft_sum = dft_sum + (xn * exp)
        result[i] = np.round(dft_sum, 9) # round to 9 decimal places and add to resultant vector

    return result # returns as a numpy array

# takes in a 2D numpy array and perform a DFT on it
def dft_2d(arr):
    # obtain rows and columns to loop over
    shape = arr.shape
    rows = shape[0]
    cols = shape[1]

    result = np.empty(shape, dtype=np.complex_)

    # get intermediate matrix
    # for each row, perform a DFT on it
    for i in range(rows):
        inter = dft(arr[i])
        result[i] = inter
    
    # perform a second DFT on transpose
    result = result.T
    for i in range(cols):
        inter = dft(result[i])
        result[i] = inter

    # transpose again to get correct shape
    result = result.T 
    
    return result




def image_convert(image_name):
    img = cv2.imread(image_name, 0)
    arr = np.array(img)
    # pad the array with zeros to the nearest power of 2
    n = 2**int(np.ceil(np.log2(max(arr.shape))))
    pad_width = [(0, n-arr.shape[0]), (0, n-arr.shape[1])]
    arr = np.pad(arr, pad_width, 'constant')
    # cv2.imshow('damn_thats_an_image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return arr


if __name__ == '__main__':
    main()


    #ABIOLA'S TESTS
    #arr = [[1.2+1j, 2.4, 3, 4], [2, 1, 0, 2]]
    # print(np.fft.fft2(arr))
    # print(fft_2d(np.asarray(arr)))


    # ks = np.array(np.arange(3))

    # n = np.array(np.arange(3))
    # kn = ks.T * n
    # first_exponens = np.exp(-1j * 2 * math.pi * kn / 3)
    # a = np.matmul(np.asarray([1, 2, 3]), first_exponens.T)
    # print(a)