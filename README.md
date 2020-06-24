# hybrid-probabilistic-matrix-factorization-model
Code for paper 'A novel hybrid deep recommendation system to differentiate user’s preference and item’s attractiveness'

## Requirements
- Python 2.7 
- Keras 0.3.3

## Training
Training can be executed with the following command:

 `python trainHaec.py`

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

True or False to preprocess raw data for ConvMF
Path to raw rating data. data format - user id::item id::rating
Path to raw item document data. item document consists of multiple text. data format - item id::text1|text2...
Users who have less than \"min_rating\" ratings will be removed (default = 1)
Maximum length of document of each item (default = 300)
Threshold to ignore terms that have a document frequency higher than the given value (default = 0.5)
Size of vocabulary (default = 8000)