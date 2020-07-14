from glob import glob
import os
import numpy as np
from scipy import io
SEED=0
np.random.seed(SEED)

def _find_diameter(n):
    if n<=100:
        return 0
    elif n<169:
        return 1
    elif n<209:
        return 2
    elif n<3000:
        return 3
    else:
        return 4
def _load_preprocessed(np_path):
    DATA = np.load('%s/data.npy'%np_path)
    train_idx = np.load('%s/train_idx.npy'%np_path)
    test_idx = np.load('%s/test_idx.npy'%np_path)
    trainY = np.load('%s/train_labels.npy'%np_path)
    testY = np.load('%s/test_labels.npy'%np_path)
    train_diameter = np.load('%s/train_diameter.npy'%np_path)
    test_diameter = np.load('%s/test_diameter.npy'%np_path)
    train_rpm = np.load('%s/train_rpm.npy'%np_path)
    test_rpm = np.load('%s/test_rpm.npy'%np_path)
    return DATA[:,:1], train_idx, test_idx, trainY, testY, train_diameter, test_diameter, train_rpm, test_rpm
def _preprocess(raw_path, out_path, frame_len=1024, frame_intv=512):
    dirnames = glob('%s/*'%raw_path)
    dirnames.sort()
    data = {}
    data['DE']=[]
    data['FE']=[]
    data['BA']=[]
    train_idcs=[]
    test_idcs=[]
    train_labels=[]
    test_labels=[]
    train_rpms=[]
    test_rpms=[]
    cnt=0
    RPM_LIST=[1797,1772,1750,1730]
    train_fault_scale=[]
    test_fault_scale=[]
    for dir in dirnames:
        l = int(os.path.basename(dir)[0])
        fnames = glob(dir+'/*.mat')
        fnames.sort()
        for f in fnames:    
            f_id = int(os.path.basename(f).split('.')[0])
            fault = _find_diameter(f_id)
            mat = io.loadmat(f)
            r=0
            s=0
            fl={}
            for m in ['DE','FE','BA']:
                fl[m]=False
            for k in mat:
                if k[-3:] == 'RPM':
                    r=mat[k][0,0]
                if k[-4:] == 'time' and k[:4] == 'X098' and 'X%03d'%f_id=='X099':
                    continue
                if k[-4:]=='time':
                    data[k[5:7]].append(mat[k])
                    fl[k[5:7]]=True
                    if s>0 and s!= mat[k].shape[0]:
                        print("This is NOT expected")

                    s=mat[k].shape[0]
            for m in ['DE','FE','BA']:
                if not fl[m]:
                    data[m].append(np.zeros((s,1)))
            idx = np.arange(cnt, cnt+s-frame_len, frame_intv)

            train_idx,test_idx = np.split(idx, [(idx.shape[0]*4)//5])
            train_idcs.append(train_idx)
            test_idcs.append(test_idx)
            train_labels.append(np.ones_like(train_idx)*l)
            test_labels.append(np.ones_like(test_idx)*l)
            train_fault_scale.append(np.ones_like(train_idx)*fault)
            test_fault_scale.append(np.ones_like(test_idx)*fault)
            if r==0:
                r=RPM_LIST[(f_id-1)%4]
            trr = np.ones(train_idx.shape[0])*r
            train_rpms.append(trr)
            test_rpms.append(np.ones_like(test_idx)*r)
            cnt+=s
    for d in data:
        data[d] = np.concatenate(data[d],axis=0)
    X=np.concatenate(list(data.values()),axis=-1)
    train_indices = np.concatenate(train_idcs)
    test_indices = np.concatenate(test_idcs)
    train_rpms = np.concatenate(train_rpms)
    test_rpms = np.concatenate(test_rpms)
    train_labels = np.concatenate(train_labels)
    test_labels = np.concatenate(test_labels)
    train_fault_scale = np.concatenate(train_fault_scale)
    test_fault_scale = np.concatenate(test_fault_scale)
    np.save('%s/data.npy'%out_path,X)
    np.save('%s/train_idx.npy'%out_path,train_indices)
    np.save('%s/test_idx.npy'%out_path,test_indices)
    np.save('%s/train_rpm.npy'%out_path,train_rpms)
    np.save('%s/test_rpm.npy'%out_path,test_rpms)
    np.save('%s/train_labels.npy'%out_path, train_labels)
    np.save('%s/test_labels.npy'%out_path, test_labels)
    np.save('%s/train_diameter.npy'%out_path, train_fault_scale)
    np.save('%s/test_diameter.npy'%out_path, test_fault_scale)
    return X[:,:1], train_indices, test_indices, train_labels, test_labels, train_fault_scale, test_fault_scale, train_rpms, test_rpms

def get_preprocessed_data(raw_path, np_path, frame_len, frame_intv):
    '''
    If data has already been preprocessed, load preprocessed data.
    else, read and preprocess data.
    '''
    npy_files = glob("%s/*.npy"%np_path)
    npy_files = [os.path.basename(x) for x in npy_files]
    if set(['data.npy','train_idx.npy','test_idx.npy','train_rpm.npy','test_rpm.npy','train_labels.npy','test_labels.npy','train_diameter.npy','test_diameter.npy']).issubset(set(npy_files)):
        return _load_preprocessed(np_path)
    else:
        return _preprocess(raw_path, np_path, frame_len, frame_intv)
        
def get_shuffle_n_mask(l, mask_params):
    train_shuffle = np.arange(l)
    np.random.shuffle(train_shuffle)
    train_shuffle, val_shuffle = np.split(train_shuffle, [4*train_shuffle.shape[0]//5])
    
    trainY = mask_params['trainY']
    train_diameter = mask_params['train_diameter']
    SCREEN_DIAM = mask_params['screen_diam']
    MASK_P = mask_params['mask_p']
    
    SEED=0
    np.random.seed(SEED)
    
    while True:
        MASK = np.array([False]*l)
        l_temp = len(MASK[(train_diameter!=SCREEN_DIAM)])
        MASK[(train_diameter!=SCREEN_DIAM)] = np.random.choice(l_temp, l_temp, replace=False) < int(MASK_P*trainY.shape[0])
        if set([0,1,2,3,4,5]).issubset(set(trainY[MASK])):
            #print('SEED : %d'%SEED)
            break
        SEED+=1
        np.random.seed(SEED)
    print("# of labeled data : %d"%MASK.sum())
    print('# of unlabeled data : %d'%(MASK.shape[0]-MASK.sum()))
    print('percentage : %.2f%%'%(100*MASK.sum()/MASK.shape[0]))
        
    show_table(MASK, trainY, train_diameter)
    return train_shuffle, val_shuffle, MASK
    
def show_table(MASK=None, Y=None, D=None, func=None, average=False):
    def count(y, d):
        if (y==0 and d != 0) or (d==0 and y!=0) or (d==2 and y>3) or (d==4 and y>2):
            return '-'
        cnt = MASK[(Y==y)&(D==d)].sum()
        return int(cnt)
    if func is None:
        func=count
    vals = [[func(j,i) for j in range(6)] for i in range(5)]
    diam = ['0"','0.007"','0.14"','0.021"','0.028"']
    print('\n')
    print('\t\t0\t1\t2\t3\t4\t5\t| %s'%('avg' if average else 'total'))
    print('='*8*9)
    for i in range(5):
        print('%s\t|'%diam[i], end='\t')
        for j in range(6):
            #print('%s'%(func(j,i)), end='\t')
            print('%s'%(str(vals[i][j])[:5]), end='\t')
        n = sum([vals[i][j] if type(vals[i][j]) in [int, float] else 0 for j in range(6)])
        if average:
            denom = sum([1 if type(vals[i][j]) in [int,float] else 0 for j in range(6)])
            
            n=n/denom
        print('| %s'%str(n)[:5])
    print('-'*8*9)
    if average:
        print('avg\t|', end='\t')
    else:
        print('total\t|', end='\t')
    ns = []
    for j in range(6):
        #print('%d'%(MASK[Y==j].sum()),end='\t')
        n = sum([vals[i][j] if type(vals[i][j]) in [int, float] else 0 for i in range(5)])
        ns.append(n)
        if average:
            denom = sum([1 if type(vals[i][j]) in [int,float] else 0 for i in range(5)])
            n=n/denom#print('%s'%str(n/denom)[:5],end='\t')
        print('%s'%str(n)[:5],end='\t') 
    n = sum(ns)
    if average:
        denom = sum([1 if type(vals[i][j]) in [int,float] else 0 for i in range(5) for j in range(6)])
        n = n/denom

    print('| %s'%str(n)[:5])
        
        