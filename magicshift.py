from math import cos, sin, sqrt, pi
from base import *
import numpy as n

def ease(t): return n.sin(t * n.pi/2)**2

def shift(path, target, shift):
    img = pngr(path)

    A = img[3]
    img = lrgb(img[:3])

    # convert
    hsv = hsv_rgb(img)

    # hue distance
    dist = anglediff(hsv[0], 2*pi*target)

    # chroma distance
    chroma = hsv[1]*hsv[2] # saturation = chroma/value

    dist = n.maximum(dist, ease(1-chroma))

    # slight blur
    dist = n.minimum(dist, gaussianblur(dist, 8, 0.75))

    # decay
    t = n.exp(-n.square(dist))

    # lerp with shifted
    hsv[0] += 2*pi*shift
    img = img = (1-t)*img + t*rgb_hsv(hsv)#n.array([t,t,t])#
    img = rgbl(img)

    pngw(path, n.array([*img, A]))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('target', type=float)
    parser.add_argument('shift', type=float)
    #parser.add_argument('power', type=float, default=0.075)
    args = parser.parse_args()

    shift(args.file, args.target, args.shift)
