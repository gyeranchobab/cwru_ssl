import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_utils import show_table

## fix seeds for reproducibility
SEED=0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True

## benchmark mode
torch.backends.cudnn.benchmark=True

def batch_step(model, X, Y, M=None, train=False, optimizer=None, step=[0,0]):
    '''
    This function runs on every batch.
    You might want to run forward pass on the model (and backpropagate to update the weights if train==True).
    
    Input:
        - model [torch.nn.Module] : Your model
        - X [torch.Tensor] : Batch of frames. size=(batch size, 1, frame length)
        - Y [torch.Tensor] : True label of the input batch. Screened labels are set as 0. size=(batch size,)
        - M [torch.Tensor] : Mask. Each element is set True for labeled data and False for unlabeled data. size=(batch size,)
        - train [bool] : Whether it is training step or evaluation step
        - optimizer [torch.optim.Optimizer] : Optimizer
        - step [List[int,int]] : [Current epoch, Current batch]
    Output:
        - [torch.Tensor] : loss of current batch
        - [torch.Tensor] : predicted labels of current batch
    '''
    
    ## if M is not given, we define M set as all True
    if M is None:
        M = torch.zeros_like(Y) == 0
        M = M.to(device=X.device, dtype=torch.bool)
        
    ## forward pass
    ## p: output of forward pass
    p = model(X)
    
    ## pred : predicted labels
    pred = torch.argmax(p, dim=1)
    
    ## class_loss : classification loss.
    class_loss = torch.tensor(0.0).to(device=X.device)
    
    ## In supervised setting, we only train model if there are labeled data.
    if M.any():
        ## compute cross entropy loss of labeled data
        class_loss += nn.CrossEntropyLoss()(p[M],Y[M])

        ## if it is training step, backpropagate and update the model weights.
        if train:
            class_loss.backward()    ## backpropagation
            optimizer.step()         ## update model weights
            optimizer.zero_grad()    ## initialize gradients
    
    return class_loss, pred
    
def epoch_step(model, data_idx, dataY, data, train=False, shuffle=None, mask=None, optimizer=None, batch_step=batch_step, device=None,batch=128,frame_len=1024, e=0, **kwargs):
    '''
    This function runs on every epoch.
    You might want to run batches on the entire dataset.
    
    Input :
        - model [torch.nn.Module] : Your model
        - data_idx [numpy.array] : Indices for your dataset (ex. train_idx, test_idx)
        - dataY [numpy.array] : True labels for your dataset (ex. trainY, testY)
        - data [numpy.array] : Whole Dataset (ex. DATA)
        - etc.
    Output : 
        - Average Loss of current epoch
        - Average Accuracy of current epoch
    '''
    ## if shuffle is not given, generate shuffle with original order.
    if shuffle is None:
        shuffle = np.arange(data_idx.shape[0])
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    ## avg_loss : average loss
    ## n : number of processed data
    ## n_correct : number of correct data
    ## n_wrong : number of wrong data
    avg_loss = 0.0
    n = 0
    n_correct = 0
    n_wrong = 0
    
    ## run batches on the entire data
    for b in range((shuffle.shape[0]-1)//batch+1):
        ## X : input batch chosen from shuffled indices. (batch size, frame length, 1)
        ## Y : true labels of input batch. (batch size,)
        X = data[data_idx[shuffle[b*batch:(b+1)*batch]][:,None]+np.arange(frame_len)]
        Y = dataY[shuffle[b*batch:(b+1)*batch]]
        
        X = np.transpose(X, [0,2,1])    ## transpose X -> (batch size, 1, frame length)
        
        ## modify X, Y from numpy.array to torch.Tensor
        X = torch.tensor(X, device=device, dtype=torch.float)
        Y = torch.tensor(Y, device=device, dtype=torch.long)
        
        ## Yall : True labels
        Yall = Y.clone()
        if mask is None:
            M = None
        else:
            ## M : mask of input batch (for semisupervised learning)
            M = mask[shuffle[b*batch:(b+1)*batch]]
            M = torch.tensor(M, device=device, dtype=torch.bool)
            
            ## set labels of masked data to 0
            Y[~M]=0
            
        ## run batch_step
        ## loss : loss of the batch
        ## pred : predicted labels of the batch
        loss, pred = batch_step(model, X, Y, train=train, optimizer=optimizer, M=M, step=[e,b], **kwargs)
        
        ## correct/wrong : # of correct/wrong prediction of the batch
        correct = (Yall[Yall==pred]).shape[0]
        wrong = (Yall[Yall!=pred]).shape[0]
        
        ## update avg_loss, n, n_correct, n_wrong
        avg_loss = (avg_loss*n + loss.item()*X.shape[0])/(n+X.shape[0])
        n += X.shape[0]
        n_correct += correct
        n_wrong += wrong
    
    return avg_loss, n_correct/(n_correct+n_wrong)

def train(model, optimizer, train_idx, trainY, data, model_name, train_shuffle, val_shuffle, mask=None, batch_step=batch_step, sav_intv=10, tol=10, sav_path = './', device=None, epoch=500, batch=128,frame_len=1024, **kwargs):
    '''
    Train model with data
    
    Input:
        - model [torch.nn.Module] : Your model
        - optimizer [torch.optim.Optimizer] : Your optimizer
        - train_idx [numpy.array] : Indices of train set
        - trainY [numpy.array] : True labels of train set
        - data [numpy.array] : Entire train dataset
        - model_name [str] : Your model name
        - train_shuffle [numpy.array] : Shuffled indices for train set
        - val_shuffle [numpy.array] : Shuffled indices for validation set
        - mask [numpy.array] : Mask for semisupervised learning
        - etc.
    '''
    
    best_acc = 0
    
    ## Iterate for maximum 'epoch' times.
    for e in range(epoch):
        timestamp = time.time()
        
        ## set training mode on the model
        model.train()
        
        ## run epoch_step
        ## train_loss : average training loss of e-th epoch
        ## train_acc : average training accuracy of e-th epoch
        train_loss, train_acc = epoch_step(model, train_idx, trainY, data, train=True, optimizer=optimizer, shuffle=train_shuffle, mask=mask, device=device, batch=batch,frame_len=frame_len,batch_step=batch_step, e=e, **kwargs)
        
        ## set evaluation mode on the model
        model.eval()
        
        with torch.no_grad():    ## with torch.no_grad(), we do not compute gradients -> faster!
            ## run epoch_step
            ## eval_loss : average validation loss of e-th epoch
            ## eval_acc : average validation accuracy of e-th epoch
            eval_loss, eval_acc = epoch_step(model, train_idx, trainY, data, train=False, shuffle=val_shuffle, mask=mask, device=device, batch=batch,frame_len=frame_len,batch_step=batch_step, e=e, **kwargs)
        
        print('(%.2fs)[Epoch %d]'%(time.time()-timestamp, e+1))
        print('\t(train) loss : %.5f,\tacc : %.5f'%(train_loss, train_acc))
        print('\t(eval) loss : %.5f,\tacc : %.5f'%(eval_loss, eval_acc))
        
        ## if current epoch result best validation accuracy, we save current weight
        if eval_acc > best_acc:
            best_acc = eval_acc    ## update best_acc
            patience = 0           ## reset patience(used for early stopping)
            torch.save(model.state_dict(), '%s/%s_best.pth'%(sav_path, model_name))    ## save current model weights
        
        ## or save model weight for every 'sav_intv' epochs
        if e%sav_intv == sav_intv-1:
            torch.save(model.state_dict(), '%s/%s_e%d.pth'%(sav_path, model_name, e+1))
            
        ## increase patience(used for early stopping)
        patience += 1
        
        ## if validation accuracy has not increase for 'tol' epochs, early stop training
        if patience > tol:
            print('Early stop at Epoch %d'%(e+1))
            break
            
def test(model, test_idx, testY, data, model_name, batch_step=batch_step, load_version='best', sav_path = './', device=None, batch=128,frame_len=1024):
    '''
    Test your model with test set.
    
    Input :
        - model [torch.nn.Module] : Your model
        - test_idx [numpy.array] : Indices of the test set (ex. test_idx)
        - testY [numpy.array] : True labels of the test set (ex. testY)
        - data [numpy.array] : Entire dataset (ex. DATA)
        - model_name [str] : model name
        - batch_step [function] : Your batch_step function
        - load_version [str] : version of weights to load (ex. 'best', 'e10', 'e20', ...)
        - etc.
    '''
    timestamp = time.time()
    
    ## load saved model
    model.load_state_dict(torch.load('%s/%s_%s.pth'%(sav_path, model_name, load_version)))
    
    ## evaluation mode
    model.eval()
    
    with torch.no_grad():  ## do not compute gradients -> fast!
        ## run epoch_step
        ## loss : Average loss of the test set
        ## acc : Average accuracy of the test set
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