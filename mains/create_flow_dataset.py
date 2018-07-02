import argparse
from data.flow.flowdata import FlowDataSet


def main():
    """
    This program requires the following data structure at the root directory:
    For Sintel:
        <directory to images>
            <set_0>
                image_<ID>.png
                ...
            <set_1>
            ...
            <set_n>
        <directory to flows>
            <set_0>
                flow_<ID>.fo
                ...
            <set_1>
            ...
            <set_n>
    For FlyingChairs:
        <data directory>
            <ID>_img1.ppm
            <ID>_img2.ppm
            <ID>_flow.flo
            ...
    """
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    data_source = FlowDataSet.SINTEL  # Sintel by default.
    if args.data_source == 'sintel':
        data_source = FlowDataSet.SINTEL
    elif args.data_source == 'flyingchairs':
        data_source = FlowDataSet.FLYING_CHAIRS
    elif args.data_source == 'flyingthings':
        data_source = FlowDataSet.FLYING_THINGS

    dataset = FlowDataSet(args.directory, validation_size=args.num_validation, data_source=data_source)
    dataset.set_verbose(True)
    dataset.preprocess_raw(args.shard_size)


def add_args(parser):
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory of the raw dataset.')
    parser.add_argument('-v', '--num_validation', type=int, default=100,
                        help='Number of data examples to use for validation.')
    parser.add_argument('-s', '--shard_size', type=int, default=25,
                        help='Maximum number of data examples in a shard.')
    parser.add_argument('-src', '--data_source', type=str, default='sintel',
                        help='Data source can be sintel, flyingchairs, or flyingthings.')


if __name__ == "__main__":
    main()
