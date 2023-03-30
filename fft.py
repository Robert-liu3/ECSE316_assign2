import sys
import cv2
import numpy as np

#parse commands
def main():
    model = 1
    filename = "images/moonlanding.png" #TODO Change this at the end to the root folder
    num_arg = len(sys.argv)
    print("number of arguments " + str(num_arg))
    
    match num_arg:
        case 5:
            model = sys.argv[2]
            filename = sys.argv[4]
            image_convert(filename)
            # print(model)
            # print(filename)
        case 3:  
            if (sys.argv[1] == "-m"):
                model = sys.argv[2]
            elif (sys.argv[2] == "-i"):
                filename = sys.argv[2]
        case _: 
            pass

#very cool turkey algo
def fft(da_list): 
    n = len(da_list)
    if n == 1:
        return da_list
    even = fft(da_list[::2])
    odd = fft(da_list[1::2])
    factor = np.exp(-2j * np.pi * np.arange(n) / n) #TODO CHANGE THIS LINE AND LINES UNDER
    return np.concatenate([even + factor[:int(n/2)] * odd,
                           even + factor[int(n/2):] * odd])



def image_convert(image_name):
    img = cv2.imread(image_name, 0)

    cv2.imshow('damn_thats_an_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


if __name__ == '__main__':
    main()