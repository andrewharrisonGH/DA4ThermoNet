#!/usr/bin/env python3

"""
    Run this script to get ThermoNet ddG predictions with forward/inverse symmetrization.
"""

from tensorflow.keras.models import load_model
import numpy as np
from argparse import ArgumentParser
import os

def parse_cmd_args():
    """
    Parse command-line arguments for running ThermoNet.

    Returns
    -------
    Parsed command-line arguments.
    """
    argparser = ArgumentParser()
    argparser.add_argument('-xf', '--features_fwd', dest='features_fwd', type=str, required=True,
                           help='Forward (WT->MT) feature tensor file.')
    argparser.add_argument('-xr', '--features_rev', dest='features_rev', type=str, required=True,
                           help='Inverse (MT->WT) feature tensor file.')
    argparser.add_argument('-m', '--model', dest='model', type=str, required=True,
                           help='HDF5 file containing trained model parameters.')
    argparser.add_argument('-o', '--output', dest='output', type=str, required=True,
                           help='File to write symmetrized predictions.')
    return argparser.parse_args()


def main():
    args = parse_cmd_args()

    # Load forward and inverse tensors
    X_fwd = np.load(args.features_fwd)
    X_rev = np.load(args.features_rev)
    X_fwd = np.moveaxis(X_fwd, 1, -1)
    X_rev = np.moveaxis(X_rev, 1, -1)

    if X_fwd.shape != X_rev.shape:
        raise ValueError("Forward and reverse tensors must have the same shape.")

    # Load trained model
    model = load_model(args.model)

    # Predict on forward and inverse tensors
    y_fwd_pred = model.predict(X_fwd)[:, 0]
    y_rev_pred = model.predict(X_rev)[:, 0]

    # Symmetrize predictions: ensures f(WT->MT) = -f(MT->WT)
    y_sym = 0.5 * (y_fwd_pred - y_rev_pred)

    # Save symmetrized predictions
    np.savetxt(fname=args.output, X=y_sym, fmt='%.3f')
    print(f"Saved symmetrized predictions to: {args.output}")


if __name__ == '__main__':
    main()