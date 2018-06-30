import argparse
import os
import tensorflow as tf
from data.interp.davis.davis_data import DavisDataSet
from context_interp.model import ContextInterp
from train.context_interp.trainer import ContextInterpTrainer


def add_args(parser):
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory of the TFRecords.')
    parser.add_argument('-v', '--validate_every', type=int, default=4000,
                        help='Defines the frequency of validation.')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='Size of the batch.')
    parser.add_argument('-c', '--checkpoint_directory', type=str,
                        help='Directory of saved checkpoints.')
    parser.add_argument('-f', '--fine_tune', dest='fine_tune',
                        action='store_true',
                        help='Whether to use fine tuning loss')
    parser.add_argument('-w', '--pwcnet_weights_path', type=str,
                        help='Path to the .npz weights for a pre-trained PWCNet.')


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

    # TODO Some of this stuff might want to go into a config json.
    config = {
        'learning_rate': 1E-4,
        'checkpoint_directory': args.checkpoint_directory,
        'fine_tune': args.fine_tune,
    }

    print('Initializing trainer...')
    trainer = ContextInterpTrainer(model, dataset, session, config)

    print('Initializing variables...')
    session.run(tf.global_variables_initializer())

    print('Loading pre-trained PWCNet...')
    model.load_pwcnet_weights(args.pwcnet_weights_path, session)
    trainer.restore()
    trainer.train(validate_every=args.validate_every)


if __name__ == "__main__":
    main()
