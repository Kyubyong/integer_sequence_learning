# coding: utf-8

from __future__ import print_function
import numpy as np
import pickle

class Hyperparams:
    batch_size = 64
    embed_dim = 300
    hidden_dim = 1000
    ctxlen = 100 # For inference
    maxlen = 200 # For training

def num2idx(num):
    """
    We're going to predict only 0 through 999.
    """
    if num == '<EMP>':
        idx = 0
    elif num == '0':
        idx = 1000    
    elif 0 < int(num) < 1000:
        idx = int(num)
    else: #<OOV>
        idx = 1001
    return idx

def idx2num(idx):
    """
    We're going to predict only 0 through 999.
    """
    if 0 < idx < 1000:
        num = str(idx)
    elif idx == 1000:
        num = '0'
    elif idx == 0:
        num = '<EMP>'
    elif idx == 1001:
        num = 'OOV'
    return num
        
def create_train_data():
    # Vectorize
    train_lines = [line.split('"')[1] for line in open('../data/train.csv', 'r').read().splitlines()[1:]]
    train_lines = train_lines[:-640] # The last 640 rows in training data is reserved for dev-set.
    test_lines = [line.split('"')[1] for line in open('../data/test.csv', 'r').read().splitlines()[1:]]
            
    xs0, xs1 = [], []
    ys0, ys1 = [], []
    for j, line in enumerate(train_lines + test_lines):
        nums = line.split(",") 
        
        # Numbers whose digits are more than 3 are excluded
        # because we are not going to predict them.
        isvalid = True
        for num in nums:
            if num.startswith("-") or len(num) > 3:
                isvalid = False
                break
        if not isvalid: continue
        
        x_ = [num2idx(num) for num in nums[-Hyperparams.maxlen:]]
        x, y = x_[:-1], x_[1:]
        
        # Bucketing
        if len(x) <= 100:
            x += [0] * (100 - len(x)) # zero postpadding
            y += [0] * (100 - len(y)) # zero postpadding
            xs0.append(x); ys0.append(y)
        else:
            x += [0] * (Hyperparams.maxlen - len(x)) # zero postpadding
            y += [0] * (Hyperparams.maxlen - len(y)) # zero postpadding
            xs1.append(x); ys1.append(y)
            
    X = [np.array(xs0), np.array(xs1)]   
    Y = [np.array(ys0), np.array(ys1)] 
    pickle.dump((X, Y), open('data/train.pkl', 'wb'), protocol=2)
    
def load_train_data(num):
    """
      Arg:
        num: int. [0, 3]
    """
    X, Y = pickle.load(open('data/train.pkl', 'rb'))
    return X[num], Y[num]

def create_val_data():
    # Vectorize
    lines = [line.split('"')[1] for line in open('../data/train.csv', 'r').read().splitlines()[1:]][-640:]
     
    xs, ys = [], []
    for line in lines:  
        nums = line.split(",") 
        x = [num2idx(num) for num in nums[-Hyperparams.ctxlen-1:-1]]
        x += [0] * (Hyperparams.ctxlen - len(x)) # zero postpadding
        
        y = nums[-1] # ground truth

        xs.append(x)
        ys.append(y)
         
    X = np.array(xs, np.int32)
    pickle.dump((X, ys), open('data/val.pkl', 'wb'), protocol=2)
    
def load_val_data():
    X, ys = pickle.load(open('data/val.pkl', 'rb'))
     
    return X, ys

def create_test_data():
    ids = [line.split(',')[0] for line in open('../data/test.csv', 'r').read().splitlines()[1:]]
    lines = [line.split('"')[1] for line in open('../data/test.csv', 'r').read().splitlines()[1:]]
    
    xs = []
    for line in lines:  
        x = [num2idx(num) for num in line.split(",")[-Hyperparams.ctxlen:]]
        x += [0] * (Hyperparams.ctxlen - len(x)) # zero postpadding
        
        xs.append(x)
    
    X = np.array(xs, np.int32)
    pickle.dump((X, ids), open('data/test.pkl', 'wb'), protocol=2)  
 
def load_test_data():
    X, ids = pickle.load(open('data/test.pkl', 'rb'))
     
    return X, ids

if __name__ == "__main__":
    create_train_data()
    create_val_data()
    create_test_data()
#     X, Y =load_train_data(3)
#     print(len(X))
    print("Done!")