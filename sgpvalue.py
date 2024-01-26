import os, sys, argparse, random, csv
import numpy as np
import pandas as pd

def main(args):
    print('lookup_url={}'.format(args.lookup_url))
    assert os.path.isfile(args.lookup_url)
    lookup_mat = pd.read_csv(args.lookup_url, sep=' ', header=None)
    lookup_mat = lookup_mat.to_numpy(dtype=float)
    print('lookup_mat={}'.format(lookup_mat.shape))

    null_scores = np.asarray([float(x) for x in args.null_scores.split(',')])
    assert len(null_scores) >= args.num_shuffle
    null_scores = null_scores[:args.num_shuffle]
    print('null_scores={}'.format(null_scores))

    observed_scores = np.asarray([float(x) for x in args.observed_scores.split(',')])
    for obv_score in observed_scores:
        assert obv_score >= 0
        s_hat = (obv_score - np.mean(null_scores)) / np.std(null_scores)
        s_hat_key = np.floor(s_hat * 1000.0) / 1000.0
        lookup_keys = lookup_mat[:, 0].flatten()
        hit_index = np.argmin(np.fabs(lookup_keys - s_hat_key))
        sg_pvalue = lookup_mat[hit_index, 1]
        print('sg_pvalue={}'.format(sg_pvalue))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--lookup_url', type=str, help='The SG look-up table')
    parser.add_argument('--observed_scores', type=str, help='The observed scores delimited by comma')
    parser.add_argument('--null_scores', type=str, help='The null scores delimited by comma')
    parser.add_argument('--num_shuffle', type=int, help='The number of null', default=50)

    args = parser.parse_args()
    main(args)


### example command
# python sgpvalue.py --lookup_url ../data/rankprop6/psiblast/pvs_f0_t39_b0.001_N1e10_m100_ls33_ss15_l0_s1.txt --null_scores '63,66,62,62,61,59,60,54,57,61,67,66,59,55,60,61,51,60,61,59' --observed_scores '620,621' --num_shuffle 20

