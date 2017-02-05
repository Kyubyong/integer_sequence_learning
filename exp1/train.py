# -*- coding: utf-8 -*-
from __future__ import print_function
from prepro import *
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout
from keras.layers import Input

def build_graph(seqlen):
    sequence = Input(shape=(seqlen,), dtype="int32")
    embedded = Embedding(13, Hyperparams.embed_dim, mask_zero=True)(sequence)
    gru1 = GRU(Hyperparams.hidden_dim, return_sequences=True)(embedded)
    after_dp = Dropout(0.5)(gru1)
    gru2 = GRU(Hyperparams.hidden_dim, return_sequences=True)(after_dp)
    after_dp = Dropout(0.5)(gru2)
    output = TimeDistributed(Dense(13, activation="softmax"))(after_dp)
    
    model = Model(input=sequence, output=output)
    
    return model

def get_the_latest_ckpt():
    import os
    import glob
    
    latest_ckpt = max(glob.glob('ckpt/*.h5'), key=os.path.getctime)
    return latest_ckpt

def main():
    digit2idx, idx2digit = load_vocab()
    _X, _Y = load_val_data()
    
    for epoch in range(0, 30):
        for subepoch in range(4):
            num = epoch % 4 # 0, 1, 2, 3
            
            # Load train data subset
            X, Y = load_train_data(num)
            Y = np.expand_dims(Y, -1)
            
            # Build model
            seqlen = X.shape[1]        
            model = build_graph(seqlen)
            
            # Load weights if necessary
            if not (epoch == 0 and subepoch == 0):
                latest_ckpt = get_the_latest_ckpt()
                model.load_weights(latest_ckpt)
            model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Train
            model.fit(X, Y, batch_size=Hyperparams.batch_size, nb_epoch=1)
            model.save_weights("ckpt/exp1-epoch-{}-{}.h5".format(epoch, subepoch))
            
        # Evaluation
        _model = build_graph(Hyperparams.ctxlen)
        _model.load_weights(latest_ckpt)

        num_hits = 0
        for step in range(0, len(_X), Hyperparams.batch_size):
            _xs = _X[step: step + Hyperparams.batch_size]
            _ys = _Y[step: step + Hyperparams.batch_size]
            _preds = []
            for _ in range(10): # We predict 10 digits at most.
                preds = _model.predict(_xs)
                preds = preds[:, -1, :] 
                preds = np.argmax(preds, axis=-1) 
                _preds.append(preds)
                preds = np.expand_dims(preds, -1) 
                _xs = np.concatenate((_xs, preds), 1)[:, 1:]
                
            _preds = np.array(_preds).transpose()
            for _pred, _y in zip(_preds, _ys):
                got = "".join(idx2digit[idx] for idx in _pred).split(",")[0]
                if got == _y:
                    num_hits += 1
        
    print("\nEpoch={}, hits={}\n".format(epoch, num_hits))              
    
if __name__ == "__main__":
    main(); print("Done")
