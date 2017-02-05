# -*- coding: utf-8 -*-
from __future__ import print_function
from prepro import *
import numpy as np
from train import build_graph, get_the_latest_ckpt

def main():
    digit2idx, idx2digit = load_vocab()
    X, I = load_test_data()
    model = build_graph(Hyperparams.ctxlen)
    latest_ckpt = get_the_latest_ckpt()
    model.load_weights(latest_ckpt)

    with open('../submission-exp1.csv', 'w') as fout:
        fout.write("Id,Last\n")
        for step in range(0, len(X), Hyperparams.batch_size):
            xs = X[step:step+Hyperparams.batch_size]
            ids = I[step:step+Hyperparams.batch_size]
            _preds = []
            for _ in range(10):
                preds = model.predict(xs)
                preds = preds[:, -1, :] #(None, 13)
                preds = np.argmax(preds, axis=-1) #(None,)
                _preds.append(preds)
                preds = np.expand_dims(preds, -1) #(None, 1)
                xs = np.concatenate((xs, preds), 1)[:, 1:]
            _preds = np.array(_preds).transpose()
            for p, id in zip(_preds, ids):
                p = "".join(idx2digit[idx] for idx in p).split(",")[0]
                fout.write("{},{}\n".format(id, p))
                fout.flush()

if __name__ == "__main__":
    main(); print("Done")
