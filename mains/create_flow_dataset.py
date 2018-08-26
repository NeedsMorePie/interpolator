import argparse
from data.flow.flowdata_preprocessor import SintelFlowDataPreprocessor, FlyingChairsFlowDataPreprocessor, FlyingThingsFlowDataPreprocessor


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

    For FlyingThings:
        frames
            TEST
            TRAIN
                <set>
                    <clip>
                        left
                            0000.png
                            ...
                        right
        optical_flow
            TEST
            TRAIN
                <set>
                    <clip>
                        into_future
                            left
                                OpticalFlowIntoFuture_0000_L.pfm
                                ...
                            right
                        into_past
    """
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    preprocessor_constructor = SintelFlowDataPreprocessor  # Sintel by default.
    if args.data_source == 'sintel':
        preprocessor_constructor = SintelFlowDataPreprocessor
    elif args.data_source == 'flyingchairs':
        preprocessor_constructor = FlyingChairsFlowDataPreprocessor
    elif args.data_source == 'flyingthings':
        preprocessor_constructor = FlyingThingsFlowDataPreprocessor

    preprocessor = preprocessor_constructor(args.directory, validation_size=args.num_validation,
                                            shard_size=args.shard_size, verbose=True)
    preprocessor.preprocess_raw()


def add_args(parser):
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory of the raw dataset.')
    parser.add_argument('-v', '--num_validation', type=int, default=100,
                        help='Number of data examples to use for validation.')
    parser.add_argument('-s', '--shard_size', type=int, default=1,
                        help='Maximum number of data examples in a shard.')
    parser.add_argument('-src', '--data_source', type=str, default='sintel',
                        help='Data source can be sintel, flyingchairs, or flyingthings.')


if __name__ == "__main__":
    main()
