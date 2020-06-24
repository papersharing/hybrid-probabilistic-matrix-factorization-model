# -*- coding: utf8 -*-
import numpy as np
import os,sys
import math
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer


# 获取用户数、电影数、职业列表
def getInfo(fileInfo,fileOccup):
    f = open(fileInfo)
    line = f.readline()
    content = line.split(' ')
    num_u = int(content[0])
    line = f.readline()
    content = line.split(' ')
    num_i = int(content[0])

    f = open(fileOccup)
    oc_list = f.readlines()
    oc_list =[oc.split('\n')[0] for oc in oc_list]

    return num_u, num_i, oc_list


# 建立所有评分的矩阵
def readR(filename,Rshape):
    Mrate = np.zeros(Rshape)
    f = open(filename, encoding='utf8')
    line = f.readline()
    while True:
        try:
            content = line.split('\t')
            u,i,r = int(content[0]),int(content[1]),int(content[2])
            Mrate[u][i] = r
            line = f.readline()
        except:
            break
    # Mrate = Mrate/5.0           # 评分归一化
    return Mrate


# 商品信息只用流派(one-hot)
def readItem1(filename,ni):
    Mitem = np.zeros((ni,20))   #1个id,19个流派
    f = open(filename, encoding='utf16')
    line = f.readline()
    while line!='':
        # content = re.split('[|,(),\n]',line)
        content = line.split('|')
        L = list()
        L.append(content[0])
        L.extend(content[5:-1])
        L.extend(content[-1].split('\n')[0])
        try:
            L = [int(i)for i in L]
        except:
            print (L[0])
        Mitem[L[0]-1] = np.array(L)
        line = f.readline()
    Mitem = Mitem[:,1:]           # 不用id
    return Mitem

# 商品特征，并转化为One-hot
def readItem(filename,ni):
    data=[]
    datadict={}
    ilist = []
    i = 0
    fe = open(filename)
    for line in fe.readlines():
        L = []
        line = line.strip().split('|')
        L.append(int(line[0]))
        L.append((line[:,5:23]))
        ilist.append(L)
        datadict['data'] = line[2]
        data.append(datadict)
    enc = preprocessing.OneHotEncoder()
    enc.fit(ilist)
    ihot = enc.transform(ilist).toarray()
    vec = DictVectorizer()
    datahot = vec.fit_transform(data).toarray()
    fe.close()
    return np.concatenate((ihot, datahot), axis=1)


def getData(fileR, fileItem, fileUser):
    R = np.load(fileR)
    mask = np.ma.masked_where(R==0,R)
    Item = np.load(fileItem)
    User = np.load(fileUser)
    return R,User,Item

def getData1(fileUser):
    User = np.load(fileUser)
    # R = np.load(fileR)
    return User


#读取评分矩阵，把index和rating分开存储
def read_rating( path):
    results = []
    if os.path.isfile(path):
        raw_ratings = open(path, 'r')
    else:
        print "Path (preprocessed) is wrong!"
        sys.exit()
    index_list = []
    rating_list = []
    all_line = raw_ratings.read().splitlines()
    for line in all_line:
        tmp = line.split()
        num_rating = int(tmp[0])
        if num_rating > 0:
            tmp_i, tmp_r = zip(*(elem.split(":") for elem in tmp[1::]))
            index_list.append(np.array(tmp_i, dtype=int))
            rating_list.append(np.array(tmp_r, dtype=float))
        else:
            index_list.append(np.array([], dtype=int))
            rating_list.append(np.array([], dtype=float))

    results.append(index_list)
    results.append(rating_list)
    return results

def readUser1(filename,nu,occup_list ):
    # 获取用户矩阵，各列id,age,gender,occupation，age和occupation是one-hot的
    no = len(occup_list)
    Muser = np.zeros((nu,(3+no)))   # nu行，id,age,gender,occupation列
    f = open(filename)
    line = f.readline()
    while line != '':
        content = line.split('|')
        L = list()
        L.append(content[0])        # id
        L.append(content[1])        # age
        if content[2] == "M":      # gender
            L.append(0)
        else:
            L.append(1)
        one_hot = [content[3].split('\n')[0]==oc for oc in occup_list]
        L.extend(one_hot)
        L = [int(s) for s in L]
        Muser[L[0] - 1] = np.array(L)
        line = f.readline()
    Muser = Muser[:,1:]             # 去掉id列
    Muser[:, 0] = preprocessing.maxabs_scale( Muser[:,0])   # age归一化
    return Muser

# 获取用户矩阵，各列id,age,gender,occupation，age和occupation，one-hot编码
def readUser(filename,nu,occup_list ):
    occdict ={}
    ulist=[]
    i=0
    fe = open(occup_list)
    for line in fe.readlines():
        line = line.strip()
        occdict[line]=i
        i=i+1
    fe.close()
    f = open(filename)
    for line in f.readlines():
        L=[]
        content = line.strip().split('|')
        L.append(content[0])
        L.append(content[1])
        if content[2]=='M':
            L.append(0)
        else:
            L.append(1)
        L.append(occdict[content[3]])
        L.append(content[4])
        ulist.append(L)
    print ulist
    enc = preprocessing.OneHotEncoder()
    enc.fit(ulist)
    ul = enc.transform(ulist).toarray()
    return ul

def data_generator(Rfilename,path,nb_batch,batch_size=None,shuffle = True):
    U = np.load('./' + path + '/User.npy',mmap_mode='r')
    I = np.load("./" + path + 'Item.npy',mmap_mode='r')
    R = np.load(Rfilename,mmap_mode='r')
    if shuffle:
        ru = np.random.permutation(U.shape[0])      # 只在第一次读的时候做shuffle
        U = U[ru,:]
        ri = np.random.permutation(I.shape[0])
        I = I[ri,:]
    else:
        ru = range(U.shape[0])
        ri = range(I.shape[0])

    if batch_size is None:
        R = R[ru,:]
        R = R[:,ri]
        batch_U = np.concatenate((R, U), axis=1)
        batch_I = np.concatenate((R.T, I), axis=1)                                  # 转置
        yield batch_U, batch_I, R
    else:
        batch = 0
        while batch <= nb_batch:
            batch_U = U[batch*batch_size:(batch+1)*batch_size]
            batch_I = I[batch*batch_size:(batch+1)*batch_size]
            batch_R_u = R[ru,:][batch*batch_size:(batch+1)*batch_size]            # 所选用户对应的评分项
            batch_R_i = R[:,ri][:,batch*batch_size:(batch+1)*batch_size]
            batch_R = batch_R_u[:,ri][:,batch*batch_size:(batch+1)*batch_size]
            batch_U = np.concatenate((batch_R_u,batch_U),axis=1)
            batch_I = np.concatenate((batch_R_i.T,batch_I),axis=1)                  # 转置
            batch += 1
            yield batch_U,batch_I,batch_R

def save_batch_data(save_path, inputU=[], inputV=[], is_New=False):
    save_name = os.path.join(sys.path[0], save_path)
    name = save_path +'U.npy'
    if not is_New:
        temp = np.load('./'+name).astype(np.float32)
        if temp.shape[0]>0:
            inputU = np.concatenate((temp,inputU), axis=0)
    np.save(save_name+'U.npy',inputU)

    name = save_path +'V.npy'
    if not is_New:
        temp = np.load('./'+name).astype(np.float32)
        if temp.shape[0]>0:
            inputV = np.concatenate((temp,inputV), axis=0)
    np.save(save_name+'V.npy',inputV)

def read_data_batch(path,batch_size=None):
    U = np.load("./" + path+'User.npy')
    I = np.load("./" + path+'Item.npy')
    R = np.load("./" + path+'R.npy')
    ru = np.random.permutation(U.shape[0])      # shuffle
    U = U[ru,:]
    batch_U = U[:batch_size]
    ri = np.random.permutation(I.shape[0])
    I = I[ri,:]
    batch_I = I[:batch_size]
    batch_R_u = R[ru,:][:batch_size]            # 所选用户对应的评分项
    batch_R_i = R[:,ri][:,:batch_size]
    batch_R = batch_R_u[:,ri][:,:batch_size]
    batch_U = np.concatenate((batch_R_u,batch_U),axis=1)
    batch_I = np.concatenate((batch_R_i.T,batch_I),axis=1)          # 转置，不知道方向有没有问题
    return batch_U,batch_I,batch_R
def shuffle_seq(rsize):
    rseq = np.random.permutation(rsize)
    newseq = np.random.randint( 0,1, size = rsize )
    for i in range(rsize):
        newseq[rseq[i]]=i
    return rseq,newseq
def read_RU_batch(R,U,i,dataset,batch_size=None):
    # U = np.load('data/ml-100k/User.npy')
    # R = np.load('data/ml-100k/R.npy')
    # ru = np.random.permutation(U.shape[0])      # shuffle
    start = (i*batch_size)% dataset
    end = min(start+batch_size,dataset)
    batch_U = U[start:end,:]
    batch_R = R[start:end,:]
    return batch_U,batch_R
def read_RU_batch1(R,U,Uorg,i,dataset,batch_size=None):
    # U = np.load('data/ml-100k/User.npy')
    # R = np.load('data/ml-100k/R.npy')
    # ru = np.random.permutation(U.shape[0])      # shuffle
    start = (i*batch_size)% dataset
    end = min(start+batch_size,dataset)
    batch_U = U[start:end,:]
    batch_R = R[start:end,:]
    batch_Uorg = Uorg[start:end,:]
    return batch_U,batch_R,batch_Uorg
#生成R1-6_train.npy，R1-6_val.npy。矩阵R[U][I]=rating。
def gen_Rnpy():
    path = './data/ml-100k/'
    #得到user、item数量，及occup职位列表
    nu, ni, occup_list = getInfo(path + 'u.info', path + 'u.occupation')
    for i in range(1,6):
        #根据u1-u6.base生成R1-6_train.npy
        R = readR(path+'u'+str(i)+'.base',(nu,ni))
        np.save(path+'R'+str(i)+'_train',R)
        #根据u1-u6.test生成R1-6_val.npy
        R = readR(path+'u'+str(i)+'.test',(nu,ni))
        np.save(path+'R'+str(i)+'_val',R)

# 生成U.npy和I.npy(用户属性、及产品属性特征),ndarray。
def gen_UIinfonpy():
    path = './data/ml-100k/'
    nu, ni, occup_list = getInfo(path + 'u.info', path + 'u.occupation')
    Item = readItem(path+"u.item",ni)
    np.save(path+'Item',Item)
    User = readUser(path+'u.user',nu,occup_list )
    np.save(path+'User',User)
    getData(path+'R.npy',path+'Item.npy',path+'User.npy')


# path = './data/ml-100k/'
# getData(path+'R.npy',path+'Item.npy',path+'User.npy')

