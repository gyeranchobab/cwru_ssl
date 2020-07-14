import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

def batch_step(model, X, Y, M=None, train=False, optimizer=None):
    if M is None:
        M = torch.zeros_like(Y) == 0
        M = M.to(device=X.device, dtype=torch.bool)
    p = model(X)
    
    pred = torch.argmax(p, dim=1)
    
    correct = (Y[Y==pred]).shape[0]
    wrong = (Y[Y!=pred]).shape[0]
    
    class_loss = torch.tensor(0.0).to(device=X.device)
    if M.any():
        class_loss += nn.CrossEntropyLoss()(p[M],Y[M])

        if train:
            class_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return class_loss, correct, wrong
    
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
        if mask is None:
            M = None
        else:
            M = mask[shuffle[b*batch:(b+1)*batch]]
            M = torch.tensor(M, device=device, dtype=torch.bool)
            Y[~M]=0
        loss, correct, wrong = batch_step(model, X, Y, train=train, optimizer=optimizer, M=M)
        avg_loss = (avg_loss*n + loss.item()*X.shape[0])/(n+X.shape[0])
        n += X.shape[0]
        n_correct += correct
        n_wrong += wrong
    return avg_loss, n_correct/(n_correct+n_wrong)

def train(model, optimizer, train_idx, trainY, data, model_name, train_shuffle, val_shuffle, mask=None, batch_step=batch_step, sav_intv=10, tol=10, sav_path = './', device=None, epoch=500, batch=128,frame_len=1024):
    best_eval = float('inf')
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
        
        if eval_loss < best_eval:
            best_eval = eval_loss
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