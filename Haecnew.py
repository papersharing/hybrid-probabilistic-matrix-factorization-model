# -*- coding: utf8 -*-
'''
Created on April 5, 2018
@author: LHJ
'''
import os
import time
from util import eval_RMSE,eval_MAE
import math
import numpy as np
import tensorflow as tf
from text_analysis.models import CNN_module
from dataUtils import data_generator, getData1,read_rating,read_RU_batch,shuffle_seq
from USDAEnew import USDAE1
# conv+sdae模型的初始化及训练
mlp_args = {
        "noise"     : 0.0,
    # "n_nodes": (512, 256, 128, 50)
        "n_nodes"   : (512,256,128,50),
        "learning_rate": .004,
        "n_epoch"  : 100,
        "data_dir": 'data/ml-1m/',
        # "batch_size": 300,
        "rho"       :0.05,
        "reg_lambda":0.01,
        "sparse_lambda":0,
        "alpha"     :0.2,
        "beta"      :10,
        # "delta"     :1,
        "save_freq":1,

}

def Haec(res_dir, train_user, train_item, valid_user, test_user,
           R, CNN_X, vocab_size, init_W=None, give_item_weight=True,
           max_iter=30, lambda_u=1, lambda_v=1, dimension=50,
           dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100):
    # explicit setting
    a = 1
    b = 0
    num_user = R.shape[0]
    num_item = R.shape[1]
    # print '=====R.all========'
    # print num_user,num_item

    fileU = '../data/pre/ml_1m/User.npy'
    # fileR = "../Data/convmf/preprocessed/movielens_100k/R.npy"

    Uinfo= getData1(fileUser=fileU)
    PREV_LOSS = 1e-50
    PREV_TE  = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    f1 = open(res_dir + '/state.log', 'w')
    # Train_R_I按用户进行汇总的评分列表[[用户1的所有评分][用户2的所有评分]...]，长6040
    Train_R_I = train_user[1]
    # Train_R_I按产品进行汇总的评分列表[[产品1的所有评分][产品2的所有评分]...]，长3544
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_item_weight is True:
        #原理：评分越多的产品，权重越大
        # item_weight:"每个产品对应的用户评论数的开方"
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        # item_weight=item_weight*(产品数/所有产品对应的评论的开方之和），类似归一化
        # 处理后，每个Item对应的权重限制在(0-5）之间，float。
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e10

    cnn_module = CNN_module(dimension, vocab_size, dropout_rate,
                            emb_dim, max_len, num_kernel_per_ws, init_W)

    # theta,cnn输出层，litst,长度3544，每个元素为长50的list，数字，取值（-0.06，0.039）。
    theta = cnn_module.get_projection_layer(CNN_X)
    V = theta
    #随机初始化前，设置种子seed，便于复现

    np.random.seed(133)
    # # # U：6040list(6040,50)，V：长为3544的list，每个list为含50个元素的子list。
    U = np.random.uniform(size=(num_user, dimension))
    # g1 = tf.Graph()
    # with tf.Session(graph=tf.get_default_graph()) as sess:
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    model = USDAE1(sess, R.shape, Uinfo.shape, is_training=True, **mlp_args)
    print("build model...")
    model.build()
    u_loss,ulatent = model.train(R.toarray(),Uinfo,U,mlp_args["learning_rate"] )
    U = ulatent
    endure_count = 5
    count = 0
    for iteration in xrange(max_iter):
        loss = 0
        mcount = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)
        #公式7的一部分，b被设置为0，所以VV为：值为lambda_u*I_k
        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)

        # 令偏导为0，得到所有的U[i]，并求U部分的loss
        #初始化U部分的loss，共计num_user（6040）个
        sub_loss = np.zeros(num_user)
        for i in xrange(num_user):
            # idx_item：用户i评论过的电影ID列表
            idx_item = train_user[0][i]
            #V_i：从V中筛选出用户i所评论的item。长度（主List）：用户i对应的评论数，子list:50.
            V_i = V[idx_item]
            #R_i用户i的所有评分,size(1,len(idx_item))
            R_i = Train_R_I[i]
            #A,size(1,50),公式7求逆的部分，每个子元素size（1,50）
            A = VV + (a - b) * (V_i.T.dot(V_i))
            # B,size(1,50),公式7后面的部分，每个子元素为一个数。
            # np.tile（A,rep）:重复rep次A来构建array；

            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)+ lambda_u * ulatent[i].T
            # linalg线性代数模块，solve（A，B），求解Ax=B 线性方程组
            U[i] = np.linalg.solve(A, B)

            # U部分的Loss
            sub_loss[i] = -0.5 * lambda_u *(np.sum(np.square(U[i]-ulatent[i])))
        loss = loss + np.sum(sub_loss)

        # 令偏导为0，得到所有的V[j]，并求V部分的loss
        # 初始化V部分的loss，共计num_item（3544）个
        sub_loss = np.zeros(num_item)
        # b为0
        UU = b * (U.T.dot(U))
        for j in xrange(num_item):
            # idx_user：产品j被评论的用户ID列表
            idx_user = train_item[0][j]
            # U_j：从U中筛选出在idx_user出现过的。长度（主List）：产品i对应的评论数，子list:50.
            U_j = U[idx_user]
            R_j = Train_R_J[j]
            # tmp_A =(U_j.T.dot(U_j))
            tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
            # A=公式8求逆的部分，多加入了item_weight[j]
            A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
            # B=公式8的后半部分，注意cnn部分loss=lambda_v * item_weight[j] * theta[j]
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                 ).sum(0) + lambda_v * item_weight[j] * theta[j]
            # print '================num_item=============', num_item
            # print len(theta[j])
            # print 'A.shape：', A.shape
            # print 'B.shape：', B.shape
            V[j] = np.linalg.solve(A, B)
            #下面三个式子：公式6的第一项，（R-UV）^2
            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])
        loss = loss + np.sum(sub_loss)

        seed = np.random.randint(100000)
        history = cnn_module.train(CNN_X, V, item_weight, seed)
        theta = cnn_module.get_projection_layer(CNN_X)
        cnn_loss = history.history['loss'][-1]
        u_loss, ulatent = model.train(R.toarray(),Uinfo,U,mlp_args["learning_rate"])

        loss = loss - 0.5 * lambda_v * cnn_loss * num_item-u_loss
        tr_eval = eval_MAE(Train_R_I, U, V, train_user[0])
        val_eval = eval_MAE(Valid_R, U, V, valid_user[0])
        te_eval = eval_MAE(Test_R, U, V, test_user[0])

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        if te_eval>PREV_TE:
            mcount += 1
        if mcount>2:
            break
        if (val_eval < pre_val_eval):
            # cnn_module.save_model(res_dir + '/CNN_weights.hdf5')
            np.savetxt(res_dir + '/U.dat', U)
            np.savetxt(res_dir + '/V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)
        else:
            count = count + 1

        pre_val_eval = val_eval

        print "Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval)
        f1.write("Iteration：%d Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
            iteration,loss, elapsed, converge, tr_eval, val_eval, te_eval))

        # if (count == endure_count):
        #     break
        PREV_LOSS = loss
        PREV_TE = te_eval
    f1.close()
