import argparse
from data.flow.flowdata import FlowDataSet


def main():
    """
    This program requires the following data structure at the root directory:
        <directory to images>
            <set_0>
            <set_1>
            ...
            <set_n>
        <directory to flows>
            <set_0>
            <set_1>
            ...
            <set_n>
    """
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    dataset = FlowDataSet(args.directory, validation_size=args.num_validation)
    dataset.set_verbose(True)
    dataset.preprocess_raw(args.shard_size)


def add_args(parser):
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory of the raw dataset.')
    parser.add_argument('-v', '--num_validation', type=int, default=100,
                        help='Number of data examples to use for validation.')
    parser.add_argument('-s', '--shard_size', type=int, default=25,
                        help='Maximum number of data examples in a shard.')


if __name__ == "__main__":
    main()
