import sys
import cv2
import numpy as np
import math

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
            print(fft_2d(image_convert(filename)))
            print(np.fft.fft2(image_convert(filename)))
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
    result = np.round(result, 9)
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

# FFT inverse, uses level parameter to ensure final result isn't scaled down too much
def fft_inverse(arr, level=0):
    n = len(arr)
    if n == 1:
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
        # obtain rows and columns to loop over
    shape = arr.shape
    rows = shape[0]
    cols = shape[1]

    result = np.empty(shape, dtype=np.complex_)

    # get intermediate matrix
    # for each row, perform a DFT on it
    for i in range(rows):
        inter = fft_inverse(arr[i])
        result[i] = inter
    
    # perform a second DFT on transpose
    result = result.T
    for i in range(cols):
        inter = fft_inverse(result[i])
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

# inverse dft
def dft_inverse(arr):
    n = arr.shape[0] # first item in tuple is the number
    result = np.empty(n, dtype=np.complex_)

    for i in range(n):
        dft_sum = 0 # reset the sum at each iteration of the loop
        for k in range(n):
            xn = arr[k]
            exp = np.exp((2j * math.pi * i * k) / n) # flip sign of complex num.
            dft_sum = dft_sum + (xn * exp)
        dft_sum = dft_sum/n # divide by n according equation
        result[i] = np.round(dft_sum, 9) # round to 9 decimal places and add to resultant vector

    return result # returns as a numpy array

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
    # main()


    #ABIOLA'S TESTS
    arr = np.random.rand(2**9, 2)
    print(np.fft.ifft2(arr))
    print(fft_2d_inverse(arr))
    # print(dft(arr))
    # print(fft(arr))


    # ks = np.array(np.arange(3))

    # n = np.array(np.arange(3))
    # kn = ks.T * n
    # first_exponens = np.exp(-1j * 2 * math.pi * kn / 3)
    # a = np.matmul(np.asarray([1, 2, 3]), first_exponens.T)
    # print(a)