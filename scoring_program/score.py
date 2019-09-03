# pylint: disable=missing-docstring, bare-except
import os
from os.path import join, isfile
import sys
import pandas as pd
from sklearn import metrics
from datetime import datetime


def mprint(msg):
    """info"""
    cur_time = datetime.now().strftime('%m-%d %H:%M:%S')
    print(f"INFO  [{cur_time}] {msg}")


def init_dirs():
    if len(sys.argv) == 1:
        # default local
        root_dir = os.getcwd()
        dirs = {
            'ref': join(root_dir, 'sample_ref'),
            'output': join(root_dir, 'sample_scoring_output'),
            'prediction': join(root_dir, 'sample_predictions')
        }

    elif len(sys.argv) == 3:
        # codalab
        dirs = {
            'ref': join(sys.argv[1], 'ref'),
            'output': sys.argv[2],
            'prediction': join(sys.argv[1], 'res')
        }

    elif len(sys.argv) == 5 and sys.argv[1] == 'local':
        # full call in local
        dirs = {
            'prediction': join(sys.argv[2]),
            'ref': join(sys.argv[3]),
            'output': sys.argv[4]
        }
    else:
        raise ValueError("Wrong number of arguments")

    os.makedirs(dirs['output'], exist_ok=True)
    return dirs


def write_score(dirs, score_file):
    datanames = sorted(os.listdir(dirs['ref']))
    mprint(f'Datanames: {datanames}')
    total_score = 0
    for idx, dataname in enumerate(datanames):
        auc = get_auc(dirs, dataname)
        total_score += auc
        score_file.write(f'set{idx+1}_score: {auc}\n')


def get_auc(dirs, dataname):
    predict_file = join(dirs['prediction'], f'{dataname}.predict')
    if not isfile(predict_file):
        mprint(f"{dataname}.predict does not exist")
        auc = 0
    else:
        prediction = pd.read_csv(predict_file)
        solution = pd.read_csv(
            join(dirs['ref'], dataname, 'main_test.solution'))
        try:
            auc = metrics.roc_auc_score(solution, prediction)
        except:
            mprint(f"{dataname}: can not caculate AUC")
            auc = 0

    mprint(f'{dataname} AUC: {auc}')
    return auc


def write_duration(dirs, score_file):
    with open(join(dirs['prediction'], 'duration.txt')) as time_f:
        time = time_f.read()
    score_file.write(f'Duration: {time}\n')


def main():
    dirs = init_dirs()
    with open(join(dirs['output'], 'scores.txt'), 'w') as score_file:
        write_score(dirs, score_file)
        write_duration(dirs, score_file)


if __name__ == '__main__':
    main()
