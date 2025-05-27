from math import cos, sin, sqrt, pi
from base import *
import numpy as n

def shift(path, shift):
    img = pngr(path)

    A = img[3]
    img = lrgb(img[:3])
    hsv = hsv_rgb(img)
    hsv[0] += 2*pi*shift
    img = rgb_hsv(hsv)

    pngw(path, n.array([*rgbl(img), A]))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('shift', type=float)
    args = parser.parse_args()

    shift(args.file, args.shift)
