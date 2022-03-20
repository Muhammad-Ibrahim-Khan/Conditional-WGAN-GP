from train import train
from controller import *

def main():
    if TRAIN:
        train(datatype=DATASET, dataset_location=DATA_LOCATION)
    else:
        print("Sorry but this project was built to build and test WGAN-GP \
              from scratch. The modules for saving, loading and testing were unnecessary "
              "and hence not implemented, thank you")


if __name__ == '__main__':
    main()
