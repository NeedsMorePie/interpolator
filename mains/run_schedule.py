import argparse
import json
from utils.misc import preprocess_var_refs


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    # Read the JSON schedule.
    print('Loading schedule...')
    with open(args.schedule) as json_data:
        schedule = json.load(json_data)
        preprocess_var_refs(schedule)
        print(schedule)
        print('')

    runs = schedule['runs']
    assert isinstance(runs, list)
    for run in runs:
        assert isinstance(run, dict)
        assert 'script' in run
        assert 'args' in run
        script = run['script']
        assert isinstance(script, str)
        args = run['args']
        assert isinstance(args, dict)


def add_args(parser):
    parser.add_argument('-s', '--schedule', type=str,
                        help='Json formatted schedule path.')


if __name__ == "__main__":
    main()
