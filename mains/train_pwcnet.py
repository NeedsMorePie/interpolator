import argparse
import json
import os
import tensorflow as tf
from data.flow.flow_data import FlowDataSet
from pwcnet.model import PWCNet
from train.pwcnet.trainer import PWCNetTrainer
from train.pwcnet.unflow_trainer import PWCNetUnflowTrainer


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True
    session = tf.Session(config=config_proto)

    # Read the JSON config.
    print('Loading configurations...')
    with open(args.config) as json_data:
        config = json.load(json_data)
        print(config)
        print('')
    # Add extra fields to the config from argparse.
    config['checkpoint_directory'] = args.checkpoint_directory
    config['directory'] = args.directory
    config['config'] = args.config

    if not os.path.exists(args.checkpoint_directory):
        os.makedirs(args.checkpoint_directory)

    print('Creating network...')
    model = PWCNet()

    print('Creating dataset...')
    dataset = FlowDataSet(args.directory, batch_size=config['batch_size'],
                          crop_size=(config['crop_height'], config['crop_width']),
                          augmentation_config=config)

    print('Initializing trainer and model ops...')
    if args.loss == 'unflow':
        trainer = PWCNetUnflowTrainer(model, dataset, session, config)
    else:
        trainer = PWCNetTrainer(model, dataset, session, config)

    print('Initializing variables...')
    session.run(tf.global_variables_initializer())
    trainer.restore()

    trainer.train(validate_every=config['validate_every'], iterations=args.iterations)


def add_args(parser):
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory of the tf records.')
    parser.add_argument('-c', '--checkpoint_directory', type=str,
                        help='Directory of saved checkpoints.')
    parser.add_argument('-j', '--config', type=str, default='mains/configs/train_pwcnet.json',
                        help='Config json file path.')
    parser.add_argument('-i', '--iterations', type=int, default=1000000,
                        help='Number of iterations to train for.')
    parser.add_argument('-l', '--loss', type=str, default='supervised',
                        help='Loss type. Can be "supervised" or "unflow". Defaults to "supervised".')


if __name__ == "__main__":
    main()
