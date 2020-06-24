# hybrid-probabilistic-matrix-factorization-model
Code for paper 'A novel hybrid deep recommendation system to differentiate user’s preference and item’s attractiveness'

## Requirements
- Python 2.7 
- Keras 0.3.3

## Training
Training can be executed with the following command:

 `python trainHaec.py`

## Configuration
The following arguments are available:

|Parameter|Default|
|-|-|
`-c`,`--do_preprocess` | `False`
`-r`,`--raw_rating_data_path` |`../data/org/ml_1m/ml-1m_ratings.dat`
`-i`,`--raw_item_document_data_path` |`../data/org/ml_plot.dat`
`-m`,`--min_rating` |`1`
`-l`,`--max_length_document` |`300`
`-f`,`--max_df` |`0.6`
`-s`,`--vocab_size` |`8000`
`-t`,`--split_ratio`|`0.6`
`-d`,`--data_path`|`../data/pre/ml_1m_40/cf/1`
`-a`,`--aux_path`|`../data/pre/ml_1m_40`
`-o`,`--res_dir`|`../data/result/ml_1m/all_mae_v1_40`
`-e`,`--emb_dim`|`Size of latent dimension for word vectors (default: 200)`
`-p`,`--pretrain_w2v`|`../data/org/glove.6B/glove.6B.200d.txt`
`-g`,`--give_item_weight`|`True`
`-k`,`dimension`|`50`
`-u`,`--lambda_u`|`10`
`-v`,`--lambda_v`|`100`
`-n`,`--max_iter`|`200`
`-w`,`--num_kernal_per_ws`|`100`

1. `do_preprocess` True or False to 
preprocess raw data for ConvMF
2. `raw_rating_data_path` Path to raw rating data. data format - user id::item id::rating
3. `raw_item_document_data_path` Path to raw item document data. item document consists of multiple text. data format - item id::text1|text2...
4. `min_rating` Users who have less than \"min_rating\" ratings will be removed (default = 1)
5. `max_length_document` Maximum length of document of each item (default = 300)
6. `max_df` Threshold to ignore terms that have a document frequency higher than the given value (default = 0.5)
7. `vocab_size`Size of vocabulary (default = 8000)
8. `split_ratio` Ratio: 1-ratio, ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and test set, respectively (default = 0.2)
9. `data_path` Path to training, valid and test data sets
10. `aux_path` Path to R, D_all sets
11. `res_dir` Path to ConvMF's result
12. `emb_dim` Size of latent dimension for word vectors (default: 200)
13. `pretrain_w2v` Path to pretrain word embedding model  to initialize word vectors
14. `give_item_weight` True or False to give item weight of ConvMF (default = False)
15. `dimension` Size of latent dimension for users and items (default: 50)
16. `lambda_u` Value of user regularizer
17. `lambda_v` Value of item regularizer
18. `max_iter` Value of max iteration (default: 200)
19. `num_kernel_per_ws` Number of kernels per window size for CNN module (default: 100)