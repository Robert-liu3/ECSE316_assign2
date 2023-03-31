import sys
import cv2
import numpy as np
import math

#parse commands
def main():
    model = 1
    filename = "moonlanding.png"
    print("hello world")
    num_arg = len(sys.argv)
    print("number of arguments " + str(num_arg))
    
    match num_arg:
        case 5:
            model = sys.argv[2]
            filename = sys.argv[4]
            mode_one(filename)
            # print(model)
            # print(filename)
        case 3:  
            if (sys.argv[1] == "-m"):
                model = sys.argv[2]
            elif (sys.argv[2] == "-i"):
                filename = sys.argv[2]
        case _: 
            pass

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

def FFT():
    pass

def mode_one(image_name):
    img = cv2.imread(image_name, 0)
    cv2.imshow('damn_thats_an_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # main()
    arr = [[1.2+1j, 2.4, 3, 4], [2, 1, 0, 2]]
    print(np.fft.fft2(arr))
    print(dft_2d(np.asarray(arr)))
    # ks = np.array(np.arange(3))

    # n = np.array(np.arange(3))
    # kn = ks.T * n
    # first_exponens = np.exp(-1j * 2 * math.pi * kn / 3)
    # a = np.matmul(np.asarray([1, 2, 3]), first_exponens.T)
    # print(a)