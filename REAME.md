# Integer Sequence Learning

[Integer Sequence Learning](https://www.kaggle.com/c/integer-sequence-learning) is one of the Kaggle competitions. Basically, you are to guess right the last number of integer sequences. The competition already ended a few months ago, but we challenge it for fun. Luckily, it's possible to check the score after the deadline. 

We apply recurrent neural networks to this task. (Why not?) This task particularly interests us as it's analogous to word prediction. That is, a number and its composing digits are equivalent to a word and characters. Based on this observation, we conduct two experiments. In the first one, we try to predict the last number based on the preceding numbers. In another one, we sequentially generate a digit based on the preceding digits. 

## Requirements

    numpy >= 1.11.1
    keras >= 1.2.1

## Task Overview

Check [this](https://www.kaggle.com/c/integer-sequence-learning) for detailed description of the task.


## Model Architecture / Hyper-parameters

 * Training
   * Inputs -> GRU Layer 1 of 1000 hidden units -> Dropout -> GRU Layer 2 of 1000 hidden units -> Dropout -> Time distributed dense -> Outputs
 * Inference
   * Inputs -> GRU Layer 1 of 1000 hidden units -> GRU Layer 2 of 1000 hidden units -> Dense -> Outputs

## Work Flow

* STEP 1. Download raw [training data](https://www.kaggle.com/c/integer-sequence-learning/download/train.csv.zip) and [test data](https://www.kaggle.com/c/integer-sequence-learning/download/test.csv.zip) and extract them to `data/` folder.
* STEP 2. Run `exp1/prepro.py` to make train/val/test data.
* STEP 3. Run `exp1/train.py`.
* STEP 4. Run `exp1/submit.py` to get the final prediction results.
* STEP 5. Run `exp2/prepro.py` to make train/val/test data.
* STEP 6. Run `exp2/train.py`.
* STEP 7. Run `exp2/submit.py` to get the final prediction results.
* STEP 8. Run `ensemble.py` to mix the results of exp1 and exp2.

### if you want to use the pretrained model,

* Download the [output files of STEP 2](https://drive.google.com/open?id=0B0ZXk88koS2KZVpfVXo3Tzd1YjA), then extract them to `exp1/data/` folder.
* Download the [pre-trained model file for exp1](https://drive.google.com/open?id=0B0ZXk88koS2KUkx2VFF2cjRuUnM), then extract it to `exp1/ckpt/` folder.
* Run `exp1/submit.py`
* Download the [output files of STEP 5](https://drive.google.com/open?id=0B0ZXk88koS2KcEZsMDIzWmlQdDg), then extract them to `exp2/data/` folder.
* Download the [pre-trained model file for exp2](https://drive.google.com/open?id=0B0ZXk88koS2KU0l1S1hTcVpwUW8), then extract it to `exp2/ckpt/` folder.
* Run `exp2/submit.py`
* Run `ensemble.py`

## Results

| Model Type | Score  |
|-- | -- | -- | 
| Exp1 | 0.13477 |
| Exp2 | 0.14557|
| Ensemble | 0.15547 |