#!/usr/bin/env python3

# import required modules
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from argparse import ArgumentParser


def parse_cmd_args():
    """
    """
    parser = ArgumentParser()
    parser.add_argument('-d', '--direct_features', dest='direct_features', type=str, required=True,
           help='Features for the training set.')
    parser.add_argument('-i', '--inverse_features', dest='inverse_features', type=str, required=True,
           help='Features for the training set.')
    parser.add_argument('-y', '--direct_targets', dest='direct_targets', type=str, required=True,
           help='Targets for the direct training set.')
    parser.add_argument('-t', '--inverse_targets', dest='inverse_targets', type=str, required=True,
           help='Targets for the inverse training set.')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, required=True,
           help='Number of training epochs.')
    parser.add_argument('--prefix', dest='prefix', type=str, required=True,
           help='Prefix for the output file to store trained model.')
    parser.add_argument('--member', required=True, type=int,
                        help='Index (1-based) of ensemble member to train')
    parser.add_argument('-k', '--k', dest='k', type=int, default=10,
                    help='Number of ensemble members.')
    args = parser.parse_args()
    return args


def build_model(model_type='regression', conv_layer_sizes=(16, 16, 16), dense_layer_size=16, dropout_rate=0.5):
    """
    """
    # make sure requested model type is valid
    if model_type not in ['regression', 'classification']:
        print('Requested model type {0} is invalid'.format(model_type))
        sys.exit(1)
        
    # instantiate a 3D convnet
    model = models.Sequential()
    model.add(layers.Conv3D(filters=conv_layer_sizes[0], kernel_size=(3, 3, 3), input_shape=(16, 16, 16, 14)))
    model.add(layers.Activation(activation='relu'))
    for c in conv_layer_sizes[1:]:
        model.add(layers.Conv3D(filters=c, kernel_size=(3, 3, 3)))
        model.add(layers.Activation(activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(units=dense_layer_size, activation='relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    
    # the last layer is dependent on model type
    if model_type == 'regression':
        model.add(layers.Dense(units=1))
    else:
        model.add(layers.Dense(units=3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.0001),
                      metrics=['accuracy'])
    
    return model


def make_dataset_memmap(X_direct, X_inverse, y_direct, y_inverse,
                        direct_idx, inverse_idx, batch_size=128, shuffle=True):
    """
    Fully vectorized input pipeline.
    Loads and processes batches of samples at once, reducing Python overhead.
    """

    # Make paired indices
    pairs = np.stack([direct_idx, inverse_idx], axis=1)
    ds = tf.data.Dataset.from_tensor_slices(pairs)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), reshuffle_each_iteration=True)

    def _load_batch(batch_pairs):
        # Already a numpy.ndarray, no need for .numpy()
        d_idx = batch_pairs[:, 0]
        i_idx = batch_pairs[:, 1]

        # Vectorized load for direct
        X_d = np.moveaxis(X_direct[d_idx], 1, -1).astype(np.float32)  # (N,14,16,16,16)->(N,16,16,16,14)
        y_d = y_direct[d_idx].astype(np.float32)

        # Vectorized load for inverse
        X_i = np.moveaxis(X_inverse[i_idx], 1, -1).astype(np.float32)
        y_i = y_inverse[i_idx].astype(np.float32)

        # Concatenate to form a full batch
        X = np.concatenate([X_d, X_i], axis=0)
        y = np.concatenate([y_d, y_i], axis=0)

        return X, y

    def _tf_load_batch(batch_pairs):
        X, y = tf.numpy_function(_load_batch, [batch_pairs], [tf.float32, tf.float32])
        X.set_shape((None, 16, 16, 16, 14))
        y.set_shape((None,))
        return X, y

    # Batch indices first, then map
    ds = ds.batch(batch_size // 2, drop_remainder=False)  # half batch of pairs → full batch after concat
    ds = ds.map(_tf_load_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def train_member(member_idx, args, X_direct, X_inverse, y_direct, y_inverse,
                 conv_layer_sizes=(16,24,32), dense_layer_size=24):
    
    batch_size = 128

    n_samples = len(y_direct)
    val_size = n_samples // args.k

    # Split indices for validation and training (balanced)
    val_direct_idx = np.arange(member_idx*val_size, (member_idx+1)*val_size)
    val_inverse_idx = val_direct_idx.copy()
    train_direct_idx = np.setdiff1d(np.arange(n_samples), val_direct_idx)
    train_inverse_idx = train_direct_idx.copy()

    # Build datasets
    train_ds = make_dataset_memmap(X_direct, X_inverse, y_direct, y_inverse,
                                   train_direct_idx, train_inverse_idx,
                                   batch_size, shuffle=True)
    val_ds = make_dataset_memmap(X_direct, X_inverse, y_direct, y_inverse,
                                 val_direct_idx, val_inverse_idx,
                                 batch_size, shuffle=False)

    model = build_model(conv_layer_sizes=conv_layer_sizes,
                        dense_layer_size=dense_layer_size)
    model.compile(loss='mse', optimizer=optimizers.Adam(1e-3), metrics=['mae'])

    model_path = f"{args.prefix}_member_{member_idx+1}.h5"
    checkpoint = callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                           save_best_only=True, mode='min')
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model.fit(train_ds, validation_data=val_ds,
              epochs=args.epochs,
              callbacks=[checkpoint, early_stopping],
              verbose=1)
    return model_path


def main():
    args = parse_cmd_args()

    # memory-mapped loading
    X_direct = np.load(args.direct_features, mmap_mode='r')
    X_inverse = np.load(args.inverse_features, mmap_mode='r')
    y_direct = np.loadtxt(args.direct_targets, dtype=np.float32)
    y_inverse = np.loadtxt(args.inverse_targets, dtype=np.float32)

    member_idx = args.member - 1
    print(f"Training ensemble member {args.member}/{args.k}")

    train_member(member_idx, args, X_direct, X_inverse, y_direct, y_inverse)

if __name__ == '__main__':
    main()
