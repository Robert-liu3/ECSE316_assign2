import sys
import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib.colors as clr

#parse commands
def main():
    model = 1
    filename = "images/moonlanding.png" # default
    num_arg = len(sys.argv)
    
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
        case 0:
            pass
        case _: 
            print("Error: invalid arguments found")
            quit()
    da_int_model = int(model)
    match da_int_model:
        case 1:
            print("Entering model 1")
            #performing fft 2d on the array
            original = cv2.imread(filename, 0) # get original copy of image for displaying and resizing later

            img_arr1 = image_convert(filename)
            fft_2d_img_1 = fft_2d(img_arr1)

            # Debugging
            # print(fft_2d_img_1)
            # print(np.fft.fft2(img_arr1))

            #convert to float
            fft_2d_img_1 = np.real(fft_2d_img_1)
            fft_2d_img_1 = cv2.resize(fft_2d_img_1, original.shape[::-1])

            #creating the graph
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            #OG image
            axs[0].imshow(img_arr1, cmap='gray')
            axs[0].set_title('Original image') 

            #2d fft image
            axs[1].imshow(np.abs(fft_2d_img_1), norm=clr.LogNorm(vmin=5), cmap='gray')
            axs[1].set_title('2D FFT LOG')
            plt.show()

        case 2:
            print("Entering mode 2")
            original = cv2.imread(filename, 0) # get original copy of image for displaying and resizing later

            # Resize image to allow FFT to work
            resized = image_convert(filename)
            # FFT first
            fft_img = fft_2d(resized.copy())

            # Set high frequencies to zero
            filter_coeff = 0.7 # fraction of values to set to zero
            n_zero = round(filter_coeff * fft_img.shape[0])

            # Flatten into a 1D array
            fft_flat = fft_img.flatten()

            # Get n indices corresponding to largest values in the multi-d array
            # Argsort puts them in the last indices
            largest_indices = fft_flat.argsort()[-n_zero:]

            # convert into index arrays to set to zero
            x_idx, y_idx = np.unravel_index(largest_indices, fft_img.shape)

            # Loop through the arrays and set each matching index (high frequencies) to 0
            for x, y, in zip(x_idx, y_idx):
                fft_img[x][y] = 0

            # Get only real portions of the transformed image, otherwise matplot throws error
            fft_2d_img_inversed = fft_2d_inverse(fft_img).real
            fft_2d_img_inversed = cv2.resize(fft_2d_img_inversed, original.shape[::-1]) # resize back to original photo's size

            # Printing non-zero coeff. info
            print("Non-zeros: " + str(int((1-filter_coeff) * fft_img.shape[0])))
            print("Fraction of non-zeros: " + str(round((1 - filter_coeff), 2)))

            # Plotting the images
            _, axis = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            axis[0].imshow(original,  # norm=LogNorm(),
                     cmap='gray',
                     interpolation='none')
            axis[0].set_title("Original")
            axis[1].imshow(fft_2d_img_inversed,  # norm=LogNorm(),
                     cmap='gray',
                     interpolation='none')
            axis[1].set_title("Denoised")

            plt.show()

        case 3:
            print("Entering mode 3")
            original = cv2.imread(filename, 0)
            #perform fft 2d on array
            img_arr3 = image_convert(filename)

            #convert to float
            fft_2d_img_3 = fft_2d(img_arr3)

            #creating the graph
            fig, axs = plt.subplots(2, 3, figsize=(10, 10))
            fig.suptitle('Compressed images at different levels')

            c_percentage = [0, 0.25, 0.5, 0.65, 0.8, 0.95]

            for i, l in enumerate(c_percentage):
                c_fft = fft_2d_img_3.copy() # copy over the transformed image each iteration

                # Similar process to mode 2 of altering
                c_fft_flat = np.abs(c_fft).flatten()

                n_zero = round(l * c_fft.shape[0]) # get percentage to remove a portion from largest indices

                to_remove = np.flip(c_fft_flat.argsort())[:n_zero] 

                x_indices, y_indices = np.unravel_index(to_remove, c_fft.shape)

                # Loop through the arrays and set each matching index to 0
                for x, y in zip(x_indices, y_indices):
                    c_fft[x][y] = 0

                # obtain non zero portions of matrix to use for generating csv files
                c_compress = c_fft[np.nonzero(c_fft)]

                np.savetxt(f"Level of compression {l*100}.csv", c_compress, delimiter=",")

                # run inverse to reconstruct the image and extract real version to display with matplot
                c_img = fft_2d_inverse(c_fft).real

                row = i // 3
                col = i % 3

                c_img = cv2.resize(c_img, original.shape[::-1]) # resize to match original image's shapes
                axs[row, col].imshow(c_img, cmap='gray')
                axs[row, col].set_title(str(l * 100) + '% Compression')
                
            plt.show()
        case 4:
            mode_4_runtimes()
        case _:
            pass

'''Naive implementations'''
def dft_naive(arr):
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
    r = 2**int(np.ceil(np.log2(img.shape[0])))
    c = 2**int(np.ceil(np.log2(img.shape[1])))

    arr = cv2.resize(img, (c, r))
    return arr

#method for mode 4

def mode_4_runtimes():
    # Define problem sizes to test
    sizes = [2**n for n in range(5, 8)] # range from 32 to 128
    
    # Create empty arrays to store mean runtimes and standard deviations
    naive_rm = np.zeros(len(sizes))
    fft_rm = np.zeros(len(sizes))
    naive_rstd = np.zeros(len(sizes))
    fft_rstd = np.zeros(len(sizes))
    
    # Iterate over problem sizes and gather runtimes
    for i, size in enumerate(sizes):
        # Generate random 2D array of size x size
        img_arr = np.random.rand(size, size)
        
        # Run naive_dft_2d 10 times and record runtimes
        naive_runtimes = []
        for j in range(10):
            start_time = time.time()
            dft_2d(img_arr)
            end_time = time.time()
            naive_runtimes.append(end_time - start_time)
        
        # Calculate mean and standard deviation of runtimes
        naive_rm[i] = np.mean(naive_runtimes)
        naive_rstd[i] = np.std(naive_runtimes)
        
        # Run fft_2d 10 times and record runtimes
        fft_runtimes = []
        for j in range(10):
            start_time = time.time()
            fft_2d(img_arr)
            end_time = time.time()
            fft_runtimes.append(end_time - start_time)
        
        # Calculate mean and standard deviation of runtimes
        fft_rm[i] = np.mean(fft_runtimes)
        fft_rstd[i] = np.std(fft_runtimes)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Runtime Graph')
    ax.errorbar(sizes, naive_rm, yerr=2*naive_rstd, label='Naive', capsize=5)
    ax.errorbar(sizes, fft_rm, yerr=2*fft_rstd, label='FFT', capsize=5)
    ax.set_xlabel('Problem Size')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
