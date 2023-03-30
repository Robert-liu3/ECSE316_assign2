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

# takes in a numpy array and performs a DFT on it
def dft(arr):
    n = arr.shape[0] # first item in tuple is the number
    result = []

    for i in range(n):
        dft_sum = 0 # reset the sum at each iteration of the loop
        for k in range(n):
            xn = arr[k]
            exp = np.exp((-2j * math.pi * i * k) / n)
            dft_sum = dft_sum + (xn * exp)
        result.append(np.ma.round(dft_sum, 9)) # round to 9 decimal places to match numpy's internal function

    return np.asarray(result) # returns as a numpy array

def FFT():
    pass

def mode_one(image_name):
    img = cv2.imread(image_name, 0)
    cv2.imshow('damn_thats_an_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # main()
    arr = [2, 1, 3]
    print(np.fft.fft(arr))
    print(dft(np.asarray(arr)))
    # ks = np.array(np.arange(3))

    # n = np.array(np.arange(3))
    # kn = ks.T * n
    # first_exponens = np.exp(-1j * 2 * math.pi * kn / 3)
    # a = np.matmul(np.asarray([1, 2, 3]), first_exponens.T)
    # print(a)