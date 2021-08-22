#!/usr/bin/env python

import argparse
import os
import utils


def main():
    description = 'Normalizes OBJ model to unit cube with centroid at 0'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('obj')
    args = parser.parse_args()
    obj = os.path.abspath(args.obj)
    objNormalized = os.path.splitext(obj)[0] + '_normalized.obj'
    utils.normalizeOBJ(obj, objNormalized)

if __name__ == '__main__':
    main()