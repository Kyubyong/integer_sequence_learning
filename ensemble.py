ids = [line.split(",")[0] for line in open('submission-exp1.csv').read().splitlines()[1:]]
exp1_preds = [line.split(",")[1] for line in open('submission-exp1.csv').read().splitlines()[1:]]
exp2_preds = [line.split(",")[1] for line in open('submission-exp2.csv').read().splitlines()[1:]]

with open("submission-ensemble.csv", 'w') as fout:
    fout.write("{},{}\n".format("Id", "Last"))
    for id, e1, e2 in zip(ids, exp1_preds, exp2_preds):
        # Replace the OOVs in exp2 with the prediction of exp1.
        pred = e2 if e2 != "OOV" else e1 
        fout.write("{},{}\n".format(id, pred))
    