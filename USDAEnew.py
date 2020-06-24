# -*- coding: utf8 -*-
'''
Created on April 5, 2018
@author: LHJ
'''
import os
import time
import tensorflow as tf
from utils import *
# from dataUtils import read_RU_batch,shuffle_seq
from dataUtils import read_RU_batch1,shuffle_seq
# 用户部分单层SDAE模型
class USDAE1(object):
    def __init__(self, sess, Rshape, Ushape, n_nodes=(20,20,20), learning_rate=0.01,
                 n_epoch=100, is_training=True, batch_size=300, decay=0.95, save_freq=1,
                 reg_lambda=0.01, rho=0.05, sparse_lambda=0.0, alpha=0.2,beta=1,delta=1, noise=0.3,
                 data_dir = None):

        self.sess = sess
        self.is_training = is_training
        self.units = n_nodes              # 隐层节点数
        self.n_layers = len(n_nodes)
        self.n_epoch = n_epoch
        self.batch_size = Rshape[0]//3
        # self.batch_size = 1000
        # self.Rshape = Rshape
        self.Unumb = Rshape[0]
        self.Rinfosize = Rshape[1]
        self.Usize = Ushape[1]
        # self.Isize = Ishape[1]
        self.lr_init = learning_rate
        self.stddev = 0.02                   # 初始化参数用的
        self.noise = noise                  # dropout水平，是数\
        self.dropout_p = 0.5                # dropout层保持概率
        self.rho = rho                      # 稀疏性系数
        self.sparse_lambda = sparse_lambda
        self.lr_decay = decay
        self.change_lr_epoch = int(n_epoch*0.3)  # 开始改变lr的epoch数
        self.regularizer = tf.contrib.layers.l2_regularizer  #使用L2正则化

        self.reg_lambda = reg_lambda        # U,V正则化系数,float
        # self.lambda_u = self.reg_lambda     # U网络权重正则化系数
        # self.lambda_i = self.reg_lambda     # I网络权重正则化系数
        self.alpha = alpha                  # U和I重建误差中，权衡S与X的系数
        self.beta = beta                    # 总loss中，U网络的重建误差权重
        # self.delta = delta                  # 总loss重，I网络的重建误差权重

        # self.summ_handle = SummaryHandle()
        # self.save_freq = save_freq          # 特征的总保存次数，每次保存n/save_freq条
        # self.save_batch_size = Rshape[0]//save_freq
        self.data_dir = data_dir

        self.checkpoint_dir = 'checkpoint'
        self.result_dir = 'results'
        self.log_dir = 'logs'
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)


    # ------------------------- 编码层 -------------------------------------
    def encoder(self, input1,input2, units, noise, layerlambda, name="encoder"):
        input1_size = int(input1.shape[1])
        input2_size = int(input2.shape[1])
        with tf.variable_scope(name):
            # mask噪声
            corrupt1 = tf.layers.dropout(input1,rate= noise,training=self.is_training)
            corrupt2 = tf.layers.dropout(input2, rate=noise, training=self.is_training)
            # 加性高斯噪声
            # corrupt = tf.add(input,noise * tf.random_uniform(input.shape))
            # 权重初始化，符合正态分布
            ew = tf.get_variable('enc_weights1', shape=[input1_size, units],
                                 initializer=tf.random_normal_initializer(mean=0.0,stddev=self.stddev),regularizer=self.regularizer(layerlambda))
            ev = tf.get_variable('enc_weights2', shape=[input2_size, units],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=self.stddev),
                                 regularizer=self.regularizer(layerlambda))
            #参数w可视化
            sew = tf.summary.histogram(name + '/enc_weights1', ew)
            sev = tf.summary.histogram(name + '/enc_weights2', ev)
            eb = tf.get_variable('enc_biases',shape=[1,units],
                                initializer=tf.constant_initializer(0.0),dtype=tf.float32,
                                regularizer=self.regularizer(layerlambda))
            seb = tf.summary.histogram(name+'/enc_biases',eb)
            # fc1 = tf.add(tf.matmul(corrupt1, ew), tf.matmul(corrupt2, ev),eb)
            fc1 = tf.add(tf.matmul(corrupt1, ew), tf.matmul(corrupt2, ev))
            fc1 = tf.add(fc1, eb)
            # fc1 = tf.layers.dropout(fc1,self.dropout_p,training=self.is_training)
            act = tf.nn.tanh(tf.layers.batch_normalization(fc1))
            # 原本是sigmoid
            # act = tf.nn.sigmoid(tf.layers.batch_normalization(fc1))
            # act = tf.nn.relu(tf.layers.batch_normalization(fc1))
            # 下面3句可视化
            self.ew = ew
            self.ev = ev
            self.eb = eb
            # self.summ_handle.add_summ(e_w=sew, e_v=sev,e_b=seb)
        return act

    # ------------------------- 译码层 -------------------------------------
    def decoder(self, input1,input2, units, layerlambda, name="decoder"):
        input1_size = int(input1.shape[1])
        input2_size = int(input2.shape[1])
        with tf.variable_scope(name):
            dw = tf.get_variable('dec_weights1', shape=[input1_size, units],
                                 initializer=tf.random_normal_initializer(mean=0.0,stddev=self.stddev),
                                 regularizer=self.regularizer(layerlambda))
            dv = tf.get_variable('dec_weights2', shape=[input2_size, units],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=self.stddev),
                                 regularizer=self.regularizer(layerlambda))
            sdw = tf.summary.histogram(name + '/dec_weights1', dw)
            sdv = tf.summary.histogram(name + '/dec_weights2', dv)
            db = tf.get_variable('dec_biases', shape=[1, units],
                                 initializer=tf.constant_initializer(0.0), dtype=tf.float32,
                                 regularizer=self.regularizer(layerlambda))
            sdb = tf.summary.histogram(name + '/dec_biases', db)
            self.dw = dw
            self.dv = dv
            self.db = db
            # self.summ_handle.add_summ(d_w=sdw,d_v=sdv, d_b=sdb)
            fc = tf.add(tf.matmul(input1, dw), tf.matmul(input2, dv))
            fc = tf.add(fc, db)
            # fc = tf.layers.dropout(fc, self.dropout_p, training=self.is_training)
            out = tf.nn.tanh(tf.layers.batch_normalization(fc))
            # 原本是sigmoid
            # out = tf.nn.sigmoid(tf.layers.batch_normalization(fc))
            # out = tf.nn.relu(tf.layers.batch_normalization(fc))
        return out

    def finaldecoder(self, input, outsize1,outsize2, layerlambda, name="decoder"):
        input_size = int(input.shape[1])
        with tf.variable_scope(name):
            dw = tf.get_variable('dec_weights1', shape=[input_size, outsize1],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=self.stddev),
                                 regularizer=self.regularizer(layerlambda))
            dv = tf.get_variable('dec_weights2', shape=[input_size, outsize2],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=self.stddev),
                                 regularizer=self.regularizer(layerlambda))
            sdw = tf.summary.histogram(name + '/dec_weights1', dw)
            sdv = tf.summary.histogram(name + '/dec_weights2', dv)
            dbs = tf.get_variable('dec_biases1', shape=[1, outsize1],
                                 initializer=tf.constant_initializer(0.0), dtype=tf.float32,
                                 regularizer=self.regularizer(layerlambda))
            dbx = tf.get_variable('dec_biases2', shape=[1, outsize2],
                                 initializer=tf.constant_initializer(0.0), dtype=tf.float32,
                                 regularizer=self.regularizer(layerlambda))
            sdbs = tf.summary.histogram(name + '/dec_biases', dbs)
            sdbx = tf.summary.histogram(name + '/dec_biases', dbx)
            self.dw = dw
            self.dv = dv
            self.dbs = dbs
            self.dbx = dbx
            # self.summ_handle.add_summ(d_w=sdw,d_v=sdv, d_bs=sdbs,d_bx=sdbx)
            fc1 = tf.add(tf.matmul(input, dw), dbs)
            fc2 = tf.add(tf.matmul(input, dv), dbx)
            out1 = tf.nn.tanh(tf.layers.batch_normalization(fc1))
            out2 = tf.nn.tanh(tf.layers.batch_normalization(fc2))
        return out1,out2

    def build(self):
        self.lr = tf.placeholder(tf.float32)
        # --------------- 用户网络 ------------------------------------------------
        # inputuinfo_size = self.Usize
        # inputur_size = self.Rinfosize
        # print '=====sdae：rsize========'
        # print self.Rinfosize
        loss_name = 'loss U'
        self.u_r = tf.placeholder(tf.float32, [None, self.Rinfosize])
        self.u_x = tf.placeholder(tf.float32, [None, self.Usize])
        self.Uorg = tf.placeholder(tf.float32, [None, self.units[-1]])
        # print "====USDAE UORG SHAPE========="
        # print self.Uorg.shape
        # ----- encoder -----
        self.U_enc_layers = []

        input1_data = self.u_r
        input2_data = self.u_x
        for i in range(self.n_layers):
            layer_name = "U_encoder_layer" + str(i)
            if i==0:
                out = self.encoder(input1_data,input2_data, self.units[i], 0, self.reg_lambda,
                                   name=layer_name)
            else:
                out = self.encoder(input1_data,input2_data, self.units[i], self.noise, self.reg_lambda,
                                  name=layer_name)
            self.U_enc_layers.append(out)

            reg_losses = tf.losses.get_regularization_losses(layer_name)
            input1_data = out
        self.U = out
        # ----- enc_reg_loss -----
        for loss in reg_losses:
            # 把变量放入一个集合，把很多变量变成一个列表
            tf.add_to_collection(loss_name, loss)

        # ----- decoder -----
        self.U_dec_layers = []
        input1_data = self.U
        input2_data = self.u_x
        # dec_nodes = list(self.units[:self.n_layers-1])        # 解码器各层节点数，与编码器对应
        dec_nodes = list(self.units[:self.n_layers-1])
        dec_nodes.reverse()
        # dec_nodes.append(input_size)
        for i in range(self.n_layers-1):
            layer_name = "U_decoder_layer" + str(i)
            out = self.decoder(input1_data,input2_data,dec_nodes[i],self.reg_lambda,layer_name)
            self.U_dec_layers.append(out)
            reg_losses = tf.losses.get_regularization_losses(layer_name)
            input1_data = out
        input = out
        out1,out2 = self.finaldecoder(input,self.Rinfosize,self.Usize,self.reg_lambda,"out_put_layer")
        Us_rec = out1
        Ux_rec = out2
        # ----- dec_reg_loss -----
        # for loss in reg_losses:
        #     tf.add_to_collection(loss_name, loss)
        # for loss in tf.losses.get_regularization_losses("out_put_layer"):
        #     tf.add_to_collection(loss_name,loss)


        # ----- total_u_loss -----


        # remember to adding！
        self.reg_losses_u = tf.add_n(tf.get_collection(loss_name))
        # self.rec_loss_u = loss_by_part(Ux_rec, self.u_x,Us_rec,self.u_r,self.alpha)
        # tf.add_to_collection(loss_name,self.rec_loss_u)

        #add_n(),实现列表里元素相加
        # self.u_loss = tf.add_n(tf.get_collection(loss_name))  # U网络重建误差
        # self.reg_losses = list(self.reg_losses_u)

        # ------------------------- 总loss ------------------------------------------------
        # self.R_hat = tf.matmul(self.U, tf.transpose(self.V))
        # self.rec_loss = mse_mask(self.R,self.R_hat)

        # tf.norm范数。
        # reg_loss_u = tf.reduce_mean(tf.norm(self.U,axis=1))
        # self.reg_losses.append(reg_loss_u)

        # the loss between sdae's output and the U.
        recon_loss = tf.reduce_sum(tf.pow(tf.subtract(self.U, self.Uorg), 2.0),axis=1)


        # the original loss
        # self.loss =  self.reg_lambda *self.reg_losses_u +self.rec_loss_u+self.beta*recon_loss
        self.loss =  self.reg_lambda *self.reg_losses_u +self.beta*recon_loss

        # print self.loss.shape
        # self.summ_handle.summ_loss = tf.summary.scalar('total loss',self.loss)
        # 输出预测准确度，和文献比一下
        # self.rmse = rmse_mask(self.R,self.R_hat)
        # self.summ_handle.summ_metric = tf.summary.scalar('rmse', self.rmse)
        # ----------------------------------------------------------------------------------

    def train(self,rinfo, uinfo,uorg,lr):
        # val_iter是进行第几次交叉验证

        # self.writer = tf.summary.FileWriter('./'+self.log_dir, self.sess.graph)

        # self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9).minimize(self.loss)
        self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        tf.global_variables_initializer().run(session=self.sess)
        # --------------------------------- 训练 ------------------------------------------
        # print 'batch_size：',self.batch_size
        n_batch = self.Unumb//self.batch_size
        # oseq, nseq = shuffle_seq(self.Unumb)
        # ru = np.random.permutation(self.Unumb)

        np.random.seed(335)
        rinfo = np.random.permutation(rinfo)
        np.random.seed(335)
        uinfo = np.random.permutation(uinfo)
        uorg = np.random.permutation(uorg)
        # Rinfo = rinfo[oseq, :]
        # Uinfo = uinfo[oseq, :]
        PREV_LOSS = 1e-50
        for epoch in range (5):
            sloss = 0
        # for i in range(n_batch):
        #     batch_U, batch_R = read_RU_batch(rinfo, uinfo, i, self.Unumb- 1, self.batch_size)
        #     _, loss, rec_loss_u,U1= self.sess.run(
        #         [self.optimizer, self.loss, self.rec_loss_u,self.U],
        #         feed_dict={self.u_r: batch_R, self.u_x: batch_U,self.lr: lr,self.Uorg:uorg})
        #
        #     loss = np.sum(loss)
        #     # converge = abs((loss - PREV_LOSS) / PREV_LOSS)
        #     # PREV_LOSS = loss
        #     sloss += loss
        #     # print "batch %d, the u_loss is:%s" % (i,loss), "Converge:%.6f" % converge
        #     # print '=====================U1============================================================='
        #     # print U1
            for i in range(n_batch):
                # batch_U, batch_R = read_RU_batch(rinfo, uinfo, i, self.Unumb - 1, self.batch_size)
                batch_U, batch_R,batch_Uorg = read_RU_batch1(rinfo, uinfo,uorg, i, self.Unumb, self.batch_size)

                # _, loss, rec_loss_u, U1 = self.sess.run([self.optimizer, self.loss, self.rec_loss_u, self.U],feed_dict={self.u_r: batch_R, self.u_x: batch_U, self.Uorg:batch_Uorg,self.lr: lr})
                _, loss, U1 = self.sess.run([self.optimizer, self.loss, self.U],feed_dict={self.u_r: batch_R, self.u_x: batch_U,self.Uorg: batch_Uorg, self.lr: lr})
                loss = np.sum(loss)
                converge = abs((loss - PREV_LOSS) / PREV_LOSS)
                PREV_LOSS = loss
                sloss += loss
                # print "batch %d, the u_loss is:%s" % (i,loss), "Converge:%.6f" % converge
                # print '=====================U1=========================================='
                # print U1
            # U1 = self.sess.run([self.U],feed_dict={self.u_r: rinfo, self.u_x: uinfo, self.lr: lr})
        U1 = self.sess.run([self.U],feed_dict={self.u_r: rinfo, self.u_x: uinfo,self.Uorg:uorg, self.lr: lr})
        return sloss, U1[0]




