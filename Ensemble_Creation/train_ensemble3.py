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
    Parses command line arguments required for training the model.
    """
    parser = ArgumentParser()
    parser.add_argument('-d', '--direct_features', dest='direct_features', type=str, required=True,
               help='Features for the training set (PRE-PROCESSED: N, D, H, W, C, float32).')
    parser.add_argument('-i', '--inverse_features', dest='inverse_features', type=str, required=True,
               help='Features for the training set (PRE-PROCESSED: N, D, H, W, C, float32).')
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


def build_model(model_type='regression', conv_layer_sizes=(16, 16, 16), dense_layer_size=16, dropout_rate=0.5, input_shape=(16, 16, 16, 14)):
    """
    Builds the 3D Convolutional Neural Network model.
    Uses the provided input_shape argument for flexibility.
    """
    # make sure requested model type is valid
    if model_type not in ['regression', 'classification']:
        print('Requested model type {0} is invalid'.format(model_type))
        sys.exit(1)
        
    # instantiate a 3D convnet
    model = models.Sequential()
    # Use the passed input_shape argument
    model.add(layers.Conv3D(filters=conv_layer_sizes[0], kernel_size=(3, 3, 3), input_shape=input_shape))
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
        # Regression output
        model.add(layers.Dense(units=1))
        # Compile is done in train_member for regression
    else:
        # Classification output
        model.add(layers.Dense(units=3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.0001),
                      metrics=['accuracy'])
    
    return model


def make_dataset_memmap(X_direct, X_inverse, y_direct, y_inverse,
                        direct_idx, inverse_idx, batch_size=128, shuffle=True):
    """
    Fully vectorized input pipeline using tf.data and memory-mapped NumPy arrays.
    
    *** IMPORTANT: This function now assumes X_direct and X_inverse are PRE-PROCESSED (N, D, H, W, C)
    and float32, removing the heavy np.moveaxis and astype calls. ***
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

        # Vectorized load for direct: ONLY SLICING (FAST)
        # Assumes X_direct is already (N, 16, 16, 16, 14) and float32
        X_d = X_direct[d_idx]
        y_d = y_direct[d_idx].astype(np.float32) # Target casting remains, but is much smaller

        # Vectorized load for inverse: ONLY SLICING (FAST)
        # Assumes X_inverse is already (N, 16, 16, 16, 14) and float32
        X_i = X_inverse[i_idx]
        y_i = y_inverse[i_idx].astype(np.float32)

        # Concatenate to form a full batch
        X = np.concatenate([X_d, X_i], axis=0)
        y = np.concatenate([y_d, y_i], axis=0)

        return X, y

    def _tf_load_batch(batch_pairs):
        # The key optimization step: loads the data using NumPy
        # Note: X is now returned as float32 directly from the sliced array
        X, y = tf.numpy_function(_load_batch, [batch_pairs], [tf.float32, tf.float32])
        
        # Explicitly set the shapes for TensorFlow graph consistency
        X.set_shape([None, 16, 16, 16, 14])
        # Reshape y to (None, 1) for explicit regression output consistency 
        y = tf.expand_dims(y, axis=-1)
        y.set_shape([None, 1]) 
        
        return X, y

    # Batch indices first, then map
    ds = ds.batch(batch_size // 2, drop_remainder=False)  # half batch of pairs → full batch after concat
    # Use AUTOTUNE to determine the optimal level of parallelism for I/O and processing
    # The CPU cores (8 of them) should now be able to keep up with the GPU
    ds = ds.map(_tf_load_batch, num_parallel_calls=8)  # Change to 'num_parallel_calls=tf.data.AUTOTUNE' if required, hardcoding to make faster.
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def train_member(member_idx, args, X_direct, X_inverse, y_direct, y_inverse,
                 conv_layer_sizes=(16,24,32), dense_layer_size=24):
    """
    Trains a single ensemble member using the defined k-fold split.
    Includes logging for data counts.
    """
    
    batch_size = 8

    n_samples = len(y_direct)
    val_size = n_samples // args.k

    # Split indices for validation and training (balanced)
    val_direct_idx = np.arange(member_idx*val_size, (member_idx+1)*val_size)
    val_inverse_idx = val_direct_idx.copy()
    train_direct_idx = np.setdiff1d(np.arange(n_samples), val_direct_idx)
    train_inverse_idx = train_direct_idx.copy()

    # Log data sizes for sanity check 
    print(f"Member {member_idx+1}: Training with {len(train_direct_idx)*2} samples, Validating with {len(val_direct_idx)*2} samples.")

    # Build datasets
    train_ds = make_dataset_memmap(X_direct, X_inverse, y_direct, y_inverse,
                                   train_direct_idx, train_inverse_idx,
                                   batch_size, shuffle=True)
    val_ds = make_dataset_memmap(X_direct, X_inverse, y_direct, y_inverse,
                                 val_direct_idx, val_inverse_idx,
                                 batch_size, shuffle=False)

    # Build model using default input shape (16, 16, 16, 14)
    model = build_model(model_type='regression', conv_layer_sizes=conv_layer_sizes, 
                dense_layer_size=dense_layer_size, dropout_rate=0.5)
    model.compile(loss='mse', optimizer=optimizers.Adam(
        lr=0.001, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=False
        ), 
        metrics=['mae']
    )

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

    # memory-mapped loading: Load the new PRE-PROCESSED files here.
    # The file names passed in the command line should correspond to the output of preprocess_features.py
    X_direct = np.load(args.direct_features, mmap_mode='r')
    X_inverse = np.load(args.inverse_features, mmap_mode='r')
    y_direct = np.loadtxt(args.direct_targets, dtype=np.float32)
    y_inverse = np.loadtxt(args.inverse_targets, dtype=np.float32)

    member_idx = args.member - 1
    print(f"Training ensemble member {args.member}/{args.k}")
    print(f"Features loaded successfully in (N, D, H, W, C) shape.")

    # Set up memory growth for GPU to avoid allocating all memory at once (best practice)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all visible GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled memory growth for GPUs.")
        except RuntimeError as e:
            # Must be set before GPUs have been initialized
            print(e)
            
    train_member(member_idx, args, X_direct, X_inverse, y_direct, y_inverse)

if __name__ == '__main__':
    main()
