import argparse
import json
import os
import subprocess
from utils.misc import compile_args, preprocess_var_refs


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    cwd = os.getcwd()
    print('CWD:', os.getcwd())

    # Read the JSON schedule.
    print('Loading schedule...')
    with open(args.schedule) as json_data:
        schedule = json.load(json_data)
        preprocess_var_refs(schedule)
        print(json.dumps(schedule, sort_keys=True, indent=4))
        print('')

    runs = schedule['runs']
    assert args.start_from < len(runs)
    assert isinstance(runs, list)
    for i in range(args.start_from, len(runs)):
        run = runs[i]
        assert isinstance(run, dict)
        assert 'executable' in run
        assert 'args' in run
        script = run['executable']
        assert isinstance(script, str)
        args_dict = run['args']
        assert isinstance(args_dict, dict)

        arg_str = ' '.join([script] + compile_args(args_dict))
        print('\nRUNNING:', arg_str, '\n')
        process = subprocess.Popen(arg_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
        while True:
            line = process.stdout.readline().rstrip()
            if not line:
                break
            print(line.decode('utf-8'))
        process.wait()


def add_args(parser):
    parser.add_argument('-s', '--schedule', type=str,
                        help='Json formatted schedule path.')
    parser.add_argument('-sf', '--start_from', type=int, default=0,
                        help='Run index to start from.')


if __name__ == "__main__":
    main()
