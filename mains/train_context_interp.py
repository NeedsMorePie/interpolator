import argparse
import os
import tensorflow as tf
from data.interp.davis.davis_data import DavisDataSet
from context_interp.model import ContextInterp
from train.context_interp.trainer import ContextInterpTrainer


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    session = tf.Session(config=config_proto)

    if not os.path.exists(args.checkpoint_directory):
        os.makedirs(args.checkpoint_directory)

    print('Creating network...')
    model = ContextInterp()

    print('Creating dataset...')
    dataset = DavisDataSet(args.directory, [[1]],
                           batch_size=args.batch_size)

    # TODO: config read from json.
    config = {
        'learning_rate': 1e-4,
        'checkpoint_directory': args.checkpoint_directory
    }

    print('Initializing trainer...')
    trainer = ContextInterpTrainer(model, dataset, session, config)

    print('Initializing variables...')
    session.run(tf.global_variables_initializer())
    # TODO: support restoring from npz dict.
    trainer.restore()

    trainer.train(validate_every=args.validate_every)


def add_args(parser):
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory of the TFRecords.')
    parser.add_argument('-v', '--validate_every', type=int, default=20,
                        help='Defines the frequency of validation.')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='Size of the batch.')
    parser.add_argument('-c', '--checkpoint_directory', type=str,
                        help='Directory of saved checkpoints.')


if __name__ == "__main__":
    main()
