# -*- coding: utf-8 -*-
from __future__ import print_function
from prepro import *
import numpy as np
from train import build_graph, get_the_latest_ckpt

def main():
    X, I = load_test_data()
    model = build_graph(Hyperparams.ctxlen)
    latest_ckpt = get_the_latest_ckpt()
    model.load_weights(latest_ckpt)
    
    with open('../submission-exp2.csv', 'w') as fout:
        fout.write("Id,Last\n")
        for step in range(0, len(X), Hyperparams.batch_size):
            xs = X[step:step+Hyperparams.batch_size]
            ids = I[step:step+Hyperparams.batch_size]
            
            preds = model.predict(xs)
            preds = preds[:, -1, :] #(None, 13)
            preds = np.argmax(preds, axis=-1) #(None,)
            for pred, id in zip(preds, ids):
                got = idx2num(pred)
                fout.write("{},{}\n".format(id, got))
                fout.flush()
                
if __name__ == "__main__":
    main(); print("Done")
