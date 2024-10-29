import os, argparse, csv, subprocess, hashlib, math, copy

import numpy as np
import pandas as pd

from Bio import SeqIO
from tqdm import tqdm

def main(args):
    ### check sg_pvalue-specific arguments
    check_sgpval_args(args)

    ### check blastp-specific arguments
    input_url, output_url, blast_args = parse_blast_args(args)
    print('input_url={}\toutput_url={}'.format(input_url, output_url))

    ### do some pre-processing
    output_dir = os.path.dirname(output_url)
    if len(output_dir) <= 0: output_dir = os.path.dirname(os.path.abspath(input_url))

    blast_argsig = blast_arg_signature(blast_args)
    print('blast_argsig={}'.format(blast_argsig))
    output_dir = os.path.join(output_dir, 'output_{}'.format(blast_argsig))
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    ### generate protein shuffles
    shuffle_url = os.path.join(output_dir, 'shuf{}_{}'.format(args.num_shuffle, os.path.basename(input_url)))
    print('shuffle_url={}'.format(shuffle_url))
    if not os.path.exists(shuffle_url) or args.overwrite > 0: generate_shuffles(input_url, shuffle_url, num_shuffle=args.num_shuffle)

    shuf_blastp_url = '{}.blastp'.format(shuffle_url)
    print('shuf_blastp_url={}'.format(shuf_blastp_url))

    ### query protein shuffles
    if not os.path.exists(shuf_blastp_url) or args.overwrite > 0:
        cmd_list = [args.blastp_url]
        shuf_blast_args = copy.deepcopy(blast_args)
        shuf_blast_args['-num_descriptions'] = 5
        shuf_blast_args['-num_alignments'] = 0
        for key in shuf_blast_args:
            cmd_list.append(key)
            cmd_list.append(str(shuf_blast_args[key]))
        cmd_list.append('-query')
        cmd_list.append(shuffle_url)
        cmd_list.append('-out')
        cmd_list.append(shuf_blastp_url)
        print('Query protein shuffles cmd={}'.format(cmd_list))
        print('Starting query protein shuffles...')
        subprocess.run(cmd_list)

    ### parse protein shuffles output
    shuf_parse_url = '{}.blastp.parsed'.format(shuf_blastp_url)
    if not os.path.exists(shuf_parse_url) or args.overwrite > 0: parse_shuffle_blastp(shuf_blastp_url, shuf_parse_url)
    query_rawbits_map, Lambda_val, K_val = load_shuffle_parse(shuf_parse_url)

    ### query input proteins
    if not os.path.exists(output_url) or args.overwrite > 0:
        cmd_list = [args.blastp_url]
        for key in blast_args:
            cmd_list.append(key)
            cmd_list.append(str(blast_args[key]))
        cmd_list.append('-query')
        cmd_list.append(input_url)
        cmd_list.append('-out')
        cmd_list.append(output_url)
        print('Query input proteins cmd={}'.format(cmd_list))
        print('Starting query input proteins...')
        subprocess.run(cmd_list)

    ### load
    print('lookup_url={}'.format(args.lookup_url))
    assert os.path.isfile(args.lookup_url)
    lookup_mat = pd.read_csv(args.lookup_url, sep=' ', header=None)
    lookup_mat = lookup_mat.to_numpy(dtype=float)
    print('lookup_mat={}'.format(lookup_mat.shape))

    ### parse protein query output
    new_blastp_url = '{}.sgpval_m{}'.format(output_url, args.num_shuffle)
    parse_blastp(output_url, new_blastp_url, query_rawbits_map, lookup_mat, Lambda_val, K_val)

def check_sgpval_args(args):
    if not os.path.isfile(args.lookup_url): assert False
    if not os.path.isfile(args.blastp_url): assert False
    assert args.num_shuffle > 0

def parse_blast_args(args):
    arg_dict = {}
    if args.db is not None: arg_dict['-db'] = args.db
    else: assert False

    if args.outfmt is not None:
        arg_dict['-outfmt'] = args.outfmt
        assert args.outfmt == 0

    if args.matrix is not None:
        arg_dict['-matrix'] = args.matrix
        assert args.matrix in ['BLOSUM45', 'BLOSUM50', 'BLOSUM62', 'BLOSUM80', 'BLOSUM90', 'PAM250', 'PAM30', 'PAM70']

    if args.evalue is not None: arg_dict['-evalue'] = args.evalue
    if args.num_descriptions is not None: arg_dict['-num_descriptions'] = args.num_descriptions
    if args.num_alignments is not None: arg_dict['-num_alignments'] = args.num_alignments
    if args.max_target_seqs is not None: arg_dict['-max_target_seqs'] = args.max_target_seqs
    if args.max_hsps is not None: arg_dict['-max_hsps'] = args.max_hsps
    if args.word_size is not None: arg_dict['-word_size'] = args.word_size
    if args.gapopen is not None: arg_dict['-gapopen'] = args.gapopen
    if args.gapextend is not None: arg_dict['-gapextend'] = args.gapextend
    if args.threshold is not None: arg_dict['-threshold'] = args.threshold
    if args.comp_based_stats is not None: arg_dict['-comp_based_stats'] = args.comp_based_stats
    if args.xdrop_gap_final is not None: arg_dict['-xdrop_gap_final'] = args.xdrop_gap_final
    if args.window_size is not None: arg_dict['-window_size'] = args.window_size

    assert args.query is not None
    input_url = args.query

    assert args.out is not None
    output_url = args.out

    return input_url, output_url, arg_dict

def blast_arg_signature(blast_args):
    arg_list = []
    for key in np.sort(list(blast_args.keys())):
        arg_list.append(key)
        arg_list.append(str(blast_args[key]))

    arg_str = ''.join(arg_list)
    print('arg_str={}'.format(arg_str))
    return hashlib.sha256(arg_str.encode()).hexdigest()

def generate_shuffles(input_url, shuffle_url, num_shuffle):
    rng = np.random.default_rng(0)
    out_file = open(shuffle_url, "w")

    records = list(SeqIO.parse(input_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        desc = record.description
        sequence = record.seq
        print('ID={}\tdesc={}\tsequence={}'.format(ID, desc, sequence))

        for shuf_idx in range(num_shuffle):
            seq_list = list(sequence)
            rng.shuffle(seq_list)
            shuffled_seq = "".join(seq_list)
            assert len(sequence) == len(shuffled_seq)

            line_num = math.ceil(len(shuffled_seq) / 60)
            out_file.write(">{}_shuffled{}\n"''.format(ID, shuf_idx))
            for i in range(line_num): out_file.write("{}\n".format(shuffled_seq[60 * i: 60 * (i + 1)]))

    out_file.close()

def parse_shuffle_blastp(shuf_blastp_url, shuf_parse_url):
    curr_query_id = ''
    curr_rawbit_list = []
    score_zone = False
    param_zone = False
    query_rawbits_map = {}

    Lambda_val = -1
    K_val = -1

    with open(shuf_blastp_url, 'r') as blast_file:
        for cnt, line in enumerate(blast_file):
            line = line.rstrip()
            line = line.rstrip("\n")
            if len(line) <= 0: continue

            if line.startswith("Query="):
                if len(curr_query_id) > 0:
                    curr_query_id = curr_query_id[:curr_query_id.rfind('_shuffled')]
                    if curr_query_id not in query_rawbits_map: query_rawbits_map[curr_query_id] = []
                    query_rawbits_map[curr_query_id].append(np.max(curr_rawbit_list))

                curr_query_id = line.split()[1]
                # print('curr_query_id={}'.format(curr_query_id))

                curr_rawbit_list = []

            elif line.startswith("Sequences"):
                score_zone = True
            elif line.startswith("Lambda"):
                score_zone = False
                param_zone = True
            elif line.startswith("Gapped"):
                Lambda_val = -1
                K_val = -1
            elif param_zone:
                arr = line.lstrip().split()
                Lambda_val = float(arr[0])
                K_val = float(arr[1])
                param_zone = False
            else:
                if not score_zone: continue

                data = line.split()
                raw_bit = float(data[-2])
                curr_rawbit_list.append(raw_bit)

        if len(curr_query_id) > 0:
            curr_query_id = curr_query_id[:curr_query_id.rfind('_shuffled')]
            if curr_query_id not in query_rawbits_map: query_rawbits_map[curr_query_id] = []
            query_rawbits_map[curr_query_id].append(np.max(curr_rawbit_list))

    out_file = open(shuf_parse_url, "w")
    out_file.write( "Lambda={}\tK={}\n".format(Lambda_val, K_val))
    print('Lambda={}\tK={}'.format(Lambda_val, K_val))

    for curr_query_id in query_rawbits_map:
        out_file.write("{}\t{}\n".format(curr_query_id, ','.join([str(x) for x in np.sort(query_rawbits_map[curr_query_id])])))
    out_file.close()

def load_shuffle_parse(shuf_parse_url):
    query_rawbits_map = {}
    Lambda_val = -1
    K_val = -1

    with open(shuf_parse_url, 'r') as parse_file:
        for cnt, line in enumerate(parse_file):
            line = line.rstrip()
            if cnt <= 0:
                Lambda_val, K_val = line.split()
                Lambda_val = float(Lambda_val[Lambda_val.rfind('=')+1:])
                K_val = float(K_val[K_val.rfind('=')+1:])
                print('Lambda={}\tK={}'.format(Lambda_val, K_val))
                continue

            curr_query_id, rawbits_str = line.split()
            rawbits_list = np.asarray(rawbits_str.split(','), dtype=float)
            # print('curr_query_id={}\trawbits_list={}'.format(curr_query_id, rawbits_list))

            assert curr_query_id not in query_rawbits_map
            query_rawbits_map[curr_query_id] = rawbits_list

    # print('query_rawbits_map={}'.format(query_rawbits_map))
    return query_rawbits_map, Lambda_val, K_val

def raw_bit_transform(bitscore, lamda, K):
    return np.round((bitscore*np.log(2)+np.log(K))/lamda)

def parse_blastp(blastp_url, new_blastp_url, query_rawbits_map, lookup_mat, Lambda_val, K_val):
    curr_query_id = ''
    score_zone = False
    header_zone = False
    out_file = open(new_blastp_url, "w")

    with open(blastp_url, 'r') as blast_file:
        for cnt, line in enumerate(blast_file):
            line = line.rstrip()
            line1 = line.lstrip()
            if len(line1) <= 0:
                out_file.write("\n")
                continue

            if line1.startswith("Query="):
                curr_query_id = line.split()[1]
                assert curr_query_id in query_rawbits_map
            elif line1.startswith("Score"):
                line = line + "\tSG"
            elif line1.startswith("Sequences"):
                score_zone = True
                line = line + "\tp-value"
            elif line1.startswith("Lambda"):
                score_zone = False
            elif score_zone:
                data = line.split()
                raw_bit = float(data[-2])
                obv_score = raw_bit_transform(raw_bit, Lambda_val, K_val)
                null_scores = raw_bit_transform(np.asarray(query_rawbits_map[curr_query_id]), Lambda_val, K_val)
                # print('rawscore={}\tnull_scores={}'.format(rawscore, null_scores))

                s_hat = (obv_score - np.mean(null_scores)) / np.std(null_scores)
                s_hat_key = np.floor(s_hat * 1000.0) / 1000.0
                lookup_keys = lookup_mat[:, 0].flatten()
                hit_index = np.argmin(np.fabs(lookup_keys - s_hat_key))
                sg_pvalue = lookup_mat[hit_index, 1]
                # print('sg_pvalue={}'.format(sg_pvalue))
                line = line + "\t{:.3f}".format(sg_pvalue)

            out_file.write("{}\n".format(line))

    out_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_shuffle', type=int, help='The number of null', default=50)
    parser.add_argument('--lookup_url', type=str, help='The SG look-up table')
    parser.add_argument('--blastp_url', type=str, help='The path to the blastp executable')
    parser.add_argument('--overwrite', type=int, help='whether to overwrite the results', default=0)

    subparsers = parser.add_subparsers(help='sub-command help')
    blast_parser = subparsers.add_parser('blastp', help='blastp')
    blast_parser.add_argument('-db', type=str, help='BLAST database name')
    blast_parser.add_argument('-query', type=str, help='Query file name')

    blast_parser.add_argument('-out', type=str, help='Output file name')
    blast_parser.add_argument('-outfmt', type=int, help='Alignment view options')

    blast_parser.add_argument('-matrix', type=str, help='Scoring matrix name')
    blast_parser.add_argument('-evalue', type=float, help='Expect value (E) for saving hits')
    blast_parser.add_argument('-num_descriptions', type=int, help='Show one-line descriptions for this number of database sequences')
    blast_parser.add_argument('-num_alignments', type=int, help='Show alignments for this number of database sequences')
    blast_parser.add_argument('-max_target_seqs', type=int, help='Number of aligned sequences to keep')
    blast_parser.add_argument('-max_hsps', type=int, help='Maximum number of HSPs (alignments) to keep for any single query-subject pair')
    blast_parser.add_argument('-word_size', type=int, help='Word size of initial match. Valid word sizes are 2-7')
    blast_parser.add_argument('-gapopen', type=int, help='Cost to open a gap')
    blast_parser.add_argument('-gapextend', type=int, help='Cost to extend a gap')

    blast_parser.add_argument('-threshold', type=int, help='Minimum score to add a word to the BLAST lookup table')
    blast_parser.add_argument('-comp_based_stats', type=str, help='Use composition-based statistics')
    blast_parser.add_argument('-xdrop_gap_final', type=float, help='Heuristic value (in bits) for final gapped alignment')
    blast_parser.add_argument('-window_size', type=int, help='Multiple hits window size, use 0 to specify 1-hit algorithm')
    # blast_parser.set_defaults(func=parse_blast_args)

    args = parser.parse_args()
    main(args)


### example command # https://www.ncbi.nlm.nih.gov/books/NBK279684/
# python blastp_wrapper.py --num_shuffle 50 --lookup_url data/rankprop6/psiblast/pvs_f0_t34_b0.001_N1e10_m50_ls33_ss15_l0_s3.txt --blastp_url /media/yanglu/Data/proj/ncbi-blast-2.14.0+-src/c++/ReleaseMT/bin/./blastp blastp -gapopen 9 -gapextend 2 -num_alignments 0 -num_descriptions 100 -evalue 10000 -matrix BLOSUM62 -db /media/yanglu/Data/proj/2019_ylu465_siglinkpred/data/rankprop6/sanity_check/db_scop_origin/scop_db -query /media/yanglu/Data/proj/2019_ylu465_siglinkpred/data/rankprop6/sanity_check/tmp.fasta -out /media/yanglu/Data/proj/2019_ylu465_siglinkpred/data/rankprop6/sanity_check/tmp.BL62.blastp
# python blastp_wrapper.py --num_shuffle 50 --lookup_url /net/noble/vol2/user/ylu465/2019_ylu465_siglinkpred/data/rankprop6/psiblast/pvs_f0_t34_b0.001_N1e10_m50_ls33_ss15_l0_s3.txt --blastp_url blastp blastp -gapopen 9 -gapextend 2 -num_alignments 0 -num_descriptions 100 -evalue 10000 -matrix BLOSUM62 -db /net/noble/vol2/user/ylu465/2019_ylu465_siglinkpred/data/rankprop6/psiblast/demo/db_scop_origin/scop_db -query /net/noble/vol2/user/ylu465/2019_ylu465_siglinkpred/data/rankprop6/psiblast/demo/example.fasta -out /net/noble/vol2/user/ylu465/2019_ylu465_siglinkpred/data/rankprop6/psiblast/demo/example.BL62.blastp
