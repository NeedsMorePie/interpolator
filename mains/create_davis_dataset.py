import argparse
import os
from data.interp.davis.davis_data import DavisDataSet


def main():
    """
    This program requires the following data structure at DAVIS/JPEGImages/{480p, 1080p}:
        <directory to shots 1>
            <video_shot_1>
            <video_shot_2>
            ...
            <video_shot_n>
        <directory to shots 2>
            <video_shot_1>
            <video_shot_2>
            ...
            <video_shot_n>
    """
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    input_directory = os.path.join(args.directory, 'JPEGImages', '480p')
    tf_records_directory = os.path.join(args.out_directory, 'tfrecords')
    dataset = DavisDataSet(tf_records_directory, [[1]])
    dataset.set_verbose(True)
    dataset.preprocess_raw(input_directory, args.shard_size, validation_size=args.num_validation)


def add_args(parser):
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory of the unzipped DAVIS dataset.')
    parser.add_argument('-o', '--out_directory', type=str,
                        help='Directory that will host the TFRecords subdirectory.')
    parser.add_argument('-v', '--num_validation', type=int, default=100,
                        help='Number of data examples to use for validation.')
    parser.add_argument('-s', '--shard_size', type=int, default=2,
                        help='Maximum number of data examples in a shard.')


if __name__ == "__main__":
    main()
