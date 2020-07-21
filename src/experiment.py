import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_utils import show_table
SEED=0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
def batch_step(model, X, Y, M=None, train=False, optimizer=None):
    if M is None:
        M = torch.zeros_like(Y) == 0
        M = M.to(device=X.device, dtype=torch.bool)
    p = model(X)
    
    pred = torch.argmax(p, dim=1)
    
    
    class_loss = torch.tensor(0.0).to(device=X.device)
    if M.any():
        class_loss += nn.CrossEntropyLoss()(p[M],Y[M])

        if train:
            class_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return class_loss, pred
    
def epoch_step(model, data_idx, dataY, data, train=False, shuffle=None, mask=None, optimizer=None, batch_step=batch_step, device=None,batch=128,frame_len=1024):
    if shuffle is None:
        shuffle = np.arange(data_idx.shape[0])
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_loss = 0.0
    n = 0
    n_correct = 0
    n_wrong = 0
    for b in range((shuffle.shape[0]-1)//batch+1):
        X = data[data_idx[shuffle[b*batch:(b+1)*batch]][:,None]+np.arange(frame_len)]
        Y = dataY[shuffle[b*batch:(b+1)*batch]]
        X = np.transpose(X, [0,2,1])
        X = torch.tensor(X, device=device, dtype=torch.float)
        Y = torch.tensor(Y, device=device, dtype=torch.long)
        Yall = Y.clone()
        if mask is None:
            M = None
        else:
            M = mask[shuffle[b*batch:(b+1)*batch]]
            M = torch.tensor(M, device=device, dtype=torch.bool)
            Y[~M]=0
        loss, pred = batch_step(model, X, Y, train=train, optimizer=optimizer, M=M)
        correct = (Yall[Yall==pred]).shape[0]
        wrong = (Yall[Yall!=pred]).shape[0]
        avg_loss = (avg_loss*n + loss.item()*X.shape[0])/(n+X.shape[0])
        n += X.shape[0]
        n_correct += correct
        n_wrong += wrong
    return avg_loss, n_correct/(n_correct+n_wrong)

def train(model, optimizer, train_idx, trainY, data, model_name, train_shuffle, val_shuffle, mask=None, batch_step=batch_step, sav_intv=10, tol=10, sav_path = './', device=None, epoch=500, batch=128,frame_len=1024):
    best_acc = 0# float('inf')
    for e in range(epoch):
        timestamp = time.time()
        
        model.train()
        train_loss, train_acc = epoch_step(model, train_idx, trainY, data, train=True, optimizer=optimizer, shuffle=train_shuffle, mask=mask, device=device, batch=batch,frame_len=frame_len,batch_step=batch_step)
        model.eval()
        with torch.no_grad():
            eval_loss, eval_acc = epoch_step(model, train_idx, trainY, data, train=False, shuffle=val_shuffle, mask=mask, device=device, batch=batch,frame_len=frame_len,batch_step=batch_step)
        
        print('(%.2fs)[Epoch %d]'%(time.time()-timestamp, e+1))
        print('\t(train) loss : %.5f,\tacc : %.5f'%(train_loss, train_acc))
        print('\t(eval) loss : %.5f,\tacc : %.5f'%(eval_loss, eval_acc))
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            patience = 0
            torch.save(model.state_dict(), '%s/%s_best.pth'%(sav_path, model_name))
        if e%sav_intv == sav_intv-1:
            torch.save(model.state_dict(), '%s/%s_e%d.pth'%(sav_path, model_name, e+1))
        patience += 1
        if patience > tol:
            print('Early stop at Epoch %d'%(e+1))
            break
            
def test(model, test_idx, testY, data, model_name, batch_step=batch_step, sav_path = './', device=None, batch=128,frame_len=1024, load_version='best'):
    timestamp = time.time()
    
    model.load_state_dict(torch.load('%s/%s_%s.pth'%(sav_path, model_name, load_version)))
    
    model.eval()
    with torch.no_grad():
        loss, acc = epoch_step(model, test_idx, testY, data, train=False, device=device, batch=batch,frame_len=frame_len,batch_step=batch_step)

    print('Test Result of model <%s>:%s'%(model_name, load_version))
    print('  [Loss]\t%.5f'%(loss))
    print('  [Accuracy]\t%.2f%%'%(acc*100))


def get_latents(model, test_idx, testY, data, model_name, load_version, sav_path, batch, frame_len, batch_step, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('%s/%s_%s.pth'%(sav_path, model_name, load_version)))
    model.eval()
    z = []
    preds = []
    with torch.no_grad():
        for b in range((test_idx.shape[0]-1)//batch+1):
            X = data[test_idx[b*batch:(b+1)*batch][:,None]+np.arange(frame_len)]
            Y = testY[b*batch:(b+1)*batch]
            X = np.transpose(X, [0,2,1])
            X = torch.tensor(X, device=device, dtype=torch.float)
            Y = torch.tensor(Y, device=device, dtype=torch.long)
            
            M = torch.zeros_like(Y)==0
            M = M.to(device=device)
            
            p_b, z_b  = model(X, get_latent=True)
            z.append(z_b)
            preds.append(torch.argmax(p_b, dim=1))
            
    z = torch.cat(z, dim=0)
    preds = torch.cat(preds, dim=0)
    return z, preds
    
def score_table(diameter, **kwargs):
    _, preds = get_latents(**kwargs)
    preds=preds.cpu().numpy()
    Y = kwargs['testY']
    correct = np.array(Y==preds)
    denoms = []
    denoms.append([len(Y[Y==i]) for i in range(5)])
    denoms.append([len(diameter[diameter==j]) for j in range(6)])
    denoms.append(len(Y))
    def score(y=None,d=None):
        if y is None:
            if d is None:
                s = correct
            else:
                s = correct[(diameter==d)]
        elif d is None:
            s = correct[(Y==y)]
        else:
            s = correct[(Y==y) & (diameter==d)]
        if len(s) == 0:
            return '-'
        else:
            return float(sum(s)/len(s))

    #vals = [[score(j,i) for j in range(6)] for i in range(5)]
    diam = ['0"','0.007"','0.14"','0.021"','0.028"']
    print('\n')
    print('\t\t0\t1\t2\t3\t4\t5\t| %s'%('total'))
    print('='*8*9)
    for i in range(5):
        print('%s\t|'%diam[i], end='\t')
        for j in range(6):
            print('%s'%(str(score(j,i))[:5]), end='\t')
        print('| %s'%str(score(d=i))[:5])
    print('-'*8*9)
    print('total\t|', end='\t')
    for j in range(6):
        print('%s'%str(score(y=j))[:5],end='\t')
    print('| %s'%str(score())[:5])
    
def confusion_matrix(**kwargs):
    _, preds = get_latents(**kwargs)
    preds = preds.cpu().numpy()
    Y = kwargs['testY']
    matrix = [[len(Y[(Y==i) & (preds==j)]) for i in range(6)] for j in range(6)]
    
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='viridis')
    ax.set_xlabel('True')
    ax.set_ylabel('Pred')
    for i in range(6):
        for j in range(6):
            text = ax.text(j, i, matrix[i][j], ha='center', va='center', color='w')
    fig.tight_layout()
    plt.show()