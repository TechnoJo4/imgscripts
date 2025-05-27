from base import *
import numpy as n

def redify(path):
    img = pngr(path)
    Z = n.zeros(n.shape(img[0]))
    pngw(path, n.array([255 - img[0], Z, Z, img[3]]))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('file', type=str)
    args = parser.parse_args()

    redify(args.file)
