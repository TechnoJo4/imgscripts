# imgscripts

My personal image editing scripts.

Usage:
- `hueshift.py <file> <shift>`: hue-shift all of the `file` by `shift`.
- `magicshift.py <file> <target> <shift>`: hue-shift saturated parts of the `file` with hue close to `target` by `shift`.
- `redify.py <file>`: remove the green and blue channels of an image, invert the red channel.

`shift` arguments are always floating-point inputs between 0 and 1.
