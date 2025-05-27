import os.path
import numpy as n
import imageio.v3 as iio
import math

# png read-write
def pngr(path):
    return iio.imread(path, mode="RGBA").transpose(2, 0, 1)

def pngw(path, data):
    f,_e = os.path.splitext(path)
    return iio.imwrite(f + ".out.png", n.clip(data, 0, 255).astype('B').transpose(1, 2, 0), mode="RGBA")

# correct matmul
def imgmmu(mat, img):
    return n.einsum("ab,axy->bxy", mat, img)

# 0-255 RGB <-> 0-1 linear sRGB
def lrgb(x: float):
    return ((1.055) * ((x/255.0)**(1.0/2.4)) - 0.055) if x>0.0 else 0
lrgb = n.vectorize(lrgb, otypes=[float])

def rgbl(x: float):
    return 255.0*((((x + 0.055)/(1 + 0.055))**2.4) if x >= 0.04045 else (x / 12.92))
rgbl = n.vectorize(rgbl, otypes=[float])

# 0-1 RGB <-> (0-2pi, 0-1, 0-1) HSV
def hsv_rgb(rgb):
    ix,iy = n.indices(n.shape(rgb[0]))
    vi = n.argmax(rgb, axis=0)
    v = n.max(rgb, axis=0)
    c = v - n.min(rgb, axis=0)

    h = n.divide(rgb[((vi+1)%3),ix,iy] - rgb[((vi+2)%3),ix,iy], c, out=n.zeros(n.shape(rgb[0])), where=(c!=0))

    return n.array([((h + 2*vi)/3 % 2) * n.pi, n.divide(c, v, n.zeros(n.shape(c)), where=v!=0), v])

def rgb_hsv(hsv):
    ix,iy = n.indices(n.shape(hsv[0]))
    c = hsv[1]*hsv[2]
    h = hsv[0]*3/n.pi
    ci2 = (n.floor(h)+1)%6
    ci = (ci2//2).astype('B')
    xi = ((ci + (2*(ci2%2) - 1)) % 3).astype('B')

    rgb = n.zeros(n.shape(hsv))
    rgb[ci,ix,iy] = c
    rgb[xi,ix,iy] = c * (1 - n.abs((h%2) - 1))

    return hsv[2]-c+rgb

# normalization
def normalize(arr):
    arr = arr - n.min(arr)
    return arr / n.max(arr)

def anglediff(x, y):
    return n.pi - n.abs(n.abs(x - y) - n.pi)

# linear sRGB <-> oklab
lms_lrgb = n.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])

oklab_lms_ = n.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
])

lms__oklab = n.array([
    [1,  0.3963377774,  0.2158037573],
    [1, -0.1055613458, -0.0638541728],
    [1, -0.0894841775, -1.2914855480],
])

lrgb_lms = n.array([
    [ 4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010],
])

def oklab_lrgb(lrgb):
    lms = imgmmu(lms_lrgb, lrgb)
    lms_ = n.cbrt(lms)
    return imgmmu(oklab_lms_, lms_)

def lrgb_oklab(oklab):
    lms_ = imgmmu(lms__oklab, oklab)
    lms = n.power(lms_, 3)
    return imgmmu(lrgb_lms, lms)

# oklab <-> oklch
def oklch_oklab(oklab):
    return n.array([
        oklab[0],
        n.sqrt(n.square(oklab[1]) + n.square(oklab[2])),
        n.arctan2(oklab[2], oklab[1])
    ])

def oklab_oklch(oklch):
    return n.array([
        oklch[0],
        oklch[1]*n.cos(oklch[2]),
        oklch[1]*n.sin(oklch[2])
    ])

# blur
def gaussian1(l, sigma):
    ax = n.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    return n.exp(-0.5 * n.square(ax) / n.square(sigma))

def gaussian2(l, sigma):
    gauss = gaussian1(l, sigma)
    return n.outer(gauss, gauss) / n.sum(kernel)

def gaussianblur(img, l, sigma):
    kernel = gaussian1(l, sigma)
    img = n.apply_along_axis(lambda x: n.convolve(x, kernel, mode='same'), 0, img)
    img = n.apply_along_axis(lambda x: n.convolve(x, kernel, mode='same'), 1, img)
    return img
