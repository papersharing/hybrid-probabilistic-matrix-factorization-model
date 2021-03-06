# -*- coding: utf8 -*-
'''
Created on April 5, 2018
@author: LHJ
'''
import tensorflow as tf
#from scipy.misc import imsave
import numpy as np

def eval_RMSE(R, U, V, TS):
    num_user = U.shape[0]
    sub_rmse = np.zeros(num_user)
    TS_count = 0
    for i in xrange(num_user):
        idx_item = TS[i]
        if len(idx_item) == 0:
            continue
        TS_count = TS_count + len(idx_item)
        approx_R_i = U[i].dot(V[idx_item].T)  # approx_R[i, idx_item]
        R_i = R[i]

        sub_rmse[i] = np.square(approx_R_i - R_i).sum()

    rmse = np.sqrt(sub_rmse.sum() / TS_count)

    return rmse

def solo_to_tuple(val, n=3):
    if type(val) in (list, tuple):
        return val
    else:
        return (val,)*n


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def mse(x,y):
    # return tf.reduce_mean(tf.pow(tf.subtract(x,y),2.0))
    return tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(x,y),2.0),axis=1))


def mse_by_part(x,y,s_size,alpha):
    assert (0 <= alpha <= 1),'alpha must set between 0 and 1.'
    # result1 = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(x[:,:s_size], y[:,:s_size]), 2.0), axis=1))
    # result2 = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(x[:,s_size:], y[:,s_size:]), 2.0), axis=1))
    #原本使用下面两行，tf.reduce_mean(A, axis=1),每行元素取平均；
    # result1 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(x[:,:s_size], y[:,:s_size]), 2.0), axis=1))
    # result2 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(x[:,s_size:], y[:,s_size:]), 2.0), axis=1))
    result1 = tf.reduce_sum(tf.pow(tf.subtract(x[:, :s_size], y[:, :s_size]), 2.0), axis=1)
    result2 = tf.reduce_sum(tf.pow(tf.subtract(x[:, s_size:], y[:, s_size:]), 2.0), axis=1)
    return alpha * result1 + (1-alpha)*result2

def loss_by_part(x,x1,y,y1,alpha):
    assert (0 <= alpha <= 1),'alpha must set between 0 and 1.'
    result1 = tf.reduce_sum(tf.pow(tf.subtract(x, x1), 2.0), axis=1)
    result2 = tf.reduce_sum(tf.pow(tf.subtract(y, y1), 2.0), axis=1)
    return alpha * result1 + (1-alpha)*result2


def mse_mask(x,y):
    # x=0的地方，不计算mse,用的sign,>0的地方置1，=0的地方置0
    mask = tf.sign(tf.abs(x))
    # return tf.reduce_mean(tf.reduce_sum(mask * tf.pow(tf.subtract(x,y),2.0)))
    return tf.reduce_mean(tf.reduce_sum(mask * tf.pow(tf.subtract(x, y), 2.0),axis=1))

def rmse_mask(x,y):
    # x=0的地方，不计算mse,用的sign,>0的地方置1，=0的地方置0
    #abs:
    mask = tf.sign(tf.abs(x))
    num = tf.reduce_sum(mask)
    mse = tf.reduce_mean(tf.reduce_sum(mask * tf.pow(tf.subtract(x, y), 2.0)))
    # return tf.sqrt(mse/num)
    return tf.sqrt(mse)

def loss_x_entropy(output, target):
    """Cross entropy loss
    Args:
        output: tensor of net output
        target: tensor of net we are trying to reconstruct
    Returns:
        Scalar tensor of cross entropy
    """
    with tf.name_scope("xentropy_loss"):
        net_output_tf = tf.convert_to_tensor(output, name='input')
        target_tf = tf.convert_to_tensor(target, name='target')
        cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'), target_tf),
                               tf.multiply(tf.log(1 - net_output_tf),(1 - target_tf)))
        return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                     name='xentropy_mean')


def sparse_loss(rho,features):
    # 其实是交叉熵
    rho_hat = tf.reduce_mean(features)
    return rho * tf.log(rho/rho_hat) +(1-rho) * tf.log((1-rho)/(1-rho_hat))


class SummaryHandle():
    def __init__(self):
        self.summ_enc_w = []
        self.summ_dec_w = []
        self.summ_enc_b = []
        self.summ_dec_b = []
        self.summ_loss = None

    def add_summ(self,e_w=None,d_w=None,e_b=None,d_b=None):
        if e_w is not None:
            self.summ_enc_w.append(e_w)
        if d_w is not None:
            self.summ_dec_w.append(d_w)
        if e_b is not None:
            self.summ_enc_b.append(e_b)
        if d_b is not None:
            self.summ_dec_b.append(d_b)