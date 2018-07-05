import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    # Read the JSON schedule.
    print('Loading schedule...')
    with open(args.schedule) as json_data:
        schedule = json.load(json_data)
        print(schedule)
        print('')


def add_args(parser):
    parser.add_argument('-s', '--schedule', type=str,
                        help='Json formatted schedule path.')


if __name__ == "__main__":
    main()
