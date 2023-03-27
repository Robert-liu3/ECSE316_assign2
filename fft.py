import sys


#parse commands
def main():
    model = 1
    print("hello world")
    num_arg = len(sys.argv)
    print("number of arguments " + str(num_arg))
    match num_arg:
        case 5:
            model = sys.argv[1]
            print(model)
        case 3:  
        case _: print("else")

    # try:
    if (sys.argv.index('-m') != None):
        index = sys.argv.index('-m') 
        mode = sys.argv[index + 1]
        print(index)
    else:
        print("bruh")
    # except:
    #     print("An exception occured")
        

if __name__ == '__main__':
    main()