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

def FFT():
    pass

def mode_one(image_name):
    img = cv2.imread(image_name, 0)
    cv2.imshow('damn_thats_an_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()