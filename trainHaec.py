# -*- coding: utf8 -*-
'''
Created on April 5, 2018
@author: LHJ
'''
import argparse
import sys
from data_manager import Data_Factory
parser = argparse.ArgumentParser()
from Haecnew import Haec
# Option for pre-processing data
parser.add_argument("-c", "--do_preprocess", type=bool,
                        help="True or False to preprocess raw data for ConvMF (default = False)", default=False)
parser.add_argument("-r", "--raw_rating_data_path", type=str,
                    help="Path to raw rating data. data format - user id::item id::rating",default='../data/org/ml_1m/ml-1m_ratings.dat')
parser.add_argument("-i", "--raw_item_document_data_path", type=str,
                    help="Path to raw item document data. item document consists of multiple text. data format - item id::text1|text2...",default='../data/org/ml_plot.dat')
parser.add_argument("-m", "--min_rating", type=int,
                    help="Users who have less than \"min_rating\" ratings will be removed (default = 1)", default=1)
parser.add_argument("-l", "--max_length_document", type=int,
                    help="Maximum length of document of each item (default = 300)", default=300)
parser.add_argument("-f", "--max_df", type=float,
                    help="Threshold to ignore terms that have a document frequency higher than the given value (default = 0.5)", default=0.6)
parser.add_argument("-s", "--vocab_size", type=int,
                    help="Size of vocabulary (default = 8000)", default=8000)
parser.add_argument("-t", "--split_ratio", type=float,
                    help="Ratio: 1-ratio, ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and test set, respectively (default = 0.2)", default=0.6)

# Option for pre-processing data and running ConvMF
parser.add_argument("-d", "--data_path", type=str,
                    help="Path to training, valid and test data sets",default='../data/pre/ml_1m_40/cf/1')
parser.add_argument("-a", "--aux_path", type=str, help="Path to R, D_all sets",default='../data/pre/ml_1m_40')

# Option for running ConvMF
parser.add_argument("-o", "--res_dir", type=str,
                    help="Path to ConvMF's result",default='../data/result/ml_1m/all_mae_v1_40')
parser.add_argument("-e", "--emb_dim", type=int,
                    help="Size of latent dimension for word vectors (default: 200)", default=200)
parser.add_argument("-p", "--pretrain_w2v", type=str,
                    help="Path to pretrain word embedding model  to initialize word vectors",default='../data/org/glove.6B/glove.6B.200d.txt')
parser.add_argument("-g", "--give_item_weight", type=bool,
                    help="True or False to give item weight of ConvMF (default = False)", default=True)
parser.add_argument("-k", "--dimension", type=int,
                    help="Size of latent dimension for users and items (default: 50)", default=50)
parser.add_argument("-u", "--lambda_u", type=float,help="Value of user regularizer",default=10)
parser.add_argument("-v", "--lambda_v", type=float,help="Value of item regularizer",default=100)
parser.add_argument("-n", "--max_iter", type=int,
                    help="Value of max iteration (default: 200)", default=200)
parser.add_argument("-w", "--num_kernel_per_ws", type=int,
                    help="Number of kernels per window size for CNN module (default: 100)", default=100)

args = parser.parse_args()
do_preprocess = args.do_preprocess
data_path = args.data_path
aux_path = args.aux_path
if data_path is None:
    sys.exit("Argument missing - data_path is required")
if aux_path is None:
    sys.exit("Argument missing - aux_path is required")

data_factory = Data_Factory()

if do_preprocess:
    path_rating = args.raw_rating_data_path
    path_itemtext = args.raw_item_document_data_path
    min_rating = args.min_rating
    max_length = args.max_length_document
    max_df = args.max_df
    vocab_size = args.vocab_size
    split_ratio = args.split_ratio

    print "=================================Preprocess Option Setting================================="
    print "\tsaving preprocessed aux path - %s" % aux_path
    print "\tsaving preprocessed data path - %s" % data_path
    print "\trating data path - %s" % path_rating
    print "\tdocument data path - %s" % path_itemtext
    print "\tmin_rating: %d\n\tmax_length_document: %d\n\tmax_df: %.1f\n\tvocab_size: %d\n\tsplit_ratio: %.1f" \
        % (min_rating, max_length, max_df, vocab_size, split_ratio)
    print "==========================================================================================="

    R, D_all = data_factory.preprocess(
        path_rating, path_itemtext, min_rating, max_length, max_df, vocab_size)
    data_factory.save(aux_path, R, D_all)
    data_factory.generate_train_valid_test_file_from_R(
        data_path, R, split_ratio)
else:
    res_dir = args.res_dir
    emb_dim = args.emb_dim
    pretrain_w2v = args.pretrain_w2v
    dimension = args.dimension
    lambda_u = args.lambda_u
    lambda_v = args.lambda_v
    max_iter = args.max_iter
    num_kernel_per_ws = args.num_kernel_per_ws
    give_item_weight = args.give_item_weight

    if res_dir is None:
        sys.exit("Argument missing - res_dir is required")
    if lambda_u is None:
        sys.exit("Argument missing - lambda_u is required")
    if lambda_v is None:
        sys.exit("Argument missing - lambda_v is required")

    print "===================================Haec Option Setting==================================="
    print "\taux path - %s" % aux_path
    print "\tdata path - %s" % data_path
    print "\tresult path - %s" % res_dir
    print "\tpretrained w2v data path - %s" % pretrain_w2v
    print "\tdimension: %d\n\tlambda_u: %.4f\n\tlambda_v: %.4f\n\tmax_iter: %d\n\tnum_kernel_per_ws: %d" \
        % (dimension, lambda_u, lambda_v, max_iter, num_kernel_per_ws)
    print "==========================================================================================="

    R, D_all = data_factory.load(aux_path)
    print
    CNN_X = D_all['X_sequence']
    vocab_size = len(D_all['X_vocab']) + 1



    if pretrain_w2v is None:
        init_W = None
    else:
        init_W = data_factory.read_pretrained_word2vec(
            pretrain_w2v, D_all['X_vocab'], emb_dim)
    #train_user[[IID列表][rating列表]],含两个元素，每个元素均为list,长度均为6040.
    train_user = data_factory.read_rating(data_path + '/train_user.dat')
    # train_item[[UID列表][rating列表]],含两个元素，每个元素均为list,长度均为3544.
    train_item = data_factory.read_rating(data_path + '/train_item.dat')

    #valid_user,test_user类似train_user，含两个元素，每个元素均为list,长度均为6040.
    valid_user = data_factory.read_rating(data_path + '/valid_user.dat')
    test_user = data_factory.read_rating(data_path + '/test_user.dat')
    Haec(max_iter=max_iter, res_dir=res_dir,
           lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, vocab_size=vocab_size, init_W=init_W, give_item_weight=give_item_weight, CNN_X=CNN_X, emb_dim=emb_dim, num_kernel_per_ws=num_kernel_per_ws,
           train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R)
