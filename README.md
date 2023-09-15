# gSASRec-pytorch

This is an official port of the gSASRec model to pytorch. The model is described in the RecSys '23 paper **"gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling"** by Aleksandr Petrov and Craig Macdonald (University of Glasgow). 

**Link to the paper:**[https://arxiv.org/pdf/2308.07192.pdf](https://arxiv.org/pdf/2308.07192.pdf)

If you use this code from the repository, please cite the work: 
```
@inproceedings{petrov2023gsasrec,
  title={gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling},
  author={Petrov, Aleksandr and Macdonald, Craig},
  booktitle={Seventeen ACM Conference on Recommender Systems},
  year={2022}
}
```


This model has been validated against the [original](https://github.com/asash/gsasrec) gSASRec tensorflow implementation.
This repository also contains a pre-split MovieLens-1M dataset and a pre-trained checkpoint of gSASRec for reproducibility. 


## Usage: 

**training**
```
python3 train_gsasrec.py --config=config_ml1m.py
```

During training, the model will print validation metrics. These metrics should not be used for reporting results. Instead, after training, run the  evaluation script: 

**evaluation**
```
python3 evaluate_gsasrec.py --config=config_ml1m.py pre_trained/gsasrec-ml1m-step:69216-t:0.75-negs:256-emb:128-dropout:0.5-metric:0.19149011481716563.pt
```
You can use a pre-trained checkpoint or use your checkpoint created by the training script. 

## Pre-Trained models
We provide checkpoints for the models for the MovieLens-1M dataset trained with gSASRec (using this config: [config_ml1m.py](config_ml1m.py)) and regular SASRec ((using this config: [config_ml1m_sasrec.py](config_ml1m_sasrec.py)))

to evaluate these models, run the following commands: 

**pre-trained SASRec model**
```python3
    python3 evaluate_gsasrec.py --config=config_ml1m_sasrec.py  --checkpoint=pre_trained/gsasrec-ml1m-step:47520-t:0.0-negs:1-emb:128-dropout:0.5-metric:0.1428058429831465.pt
```
**pre-trained gSASRec model**
```python3
  python3 evaluate_gsasrec.py --config=config_ml1m.py  --checkpoint pre_trained/gsasrec-ml1m-step:86064-t:0.75-negs:256-emb:128-dropout:0.5-metric:0.1974453226738962.pt
```

Evaluation results for the pre-trained models: 

| Model   | Loss | Number of Negatives   per Positive | t    | Recall@1 | Recall@10 | NDCG@10 |
| ------- | ---- | ---------------------------------- | ---- | -------- | --------- | ------- |
| SASrec  | BCE  | 1                                  | 0.0  | 0.0442   |    0.2392       |  0.1259       |
| gSASRec | gBCE | 128                                | 0.75 |      0.0754     |    0.3003       |   0.1726      |





## gSASrec and gBCE info info
**gSASRec** is a SASRec-based sequential recommendation model that utilises more negatives per positive and gBCE loss: 

```math
\begin{align}
     \mathcal{L}^{\beta}_{gBCE} = -\frac{1}{|I_k^-| + 1} \left( \log(\sigma^{\beta}(s_{i^+})) + \sum_{i \in I_k^{-}}\log(1-\sigma(s_i)) \right)
\end{align}
```
where $`i^+`$ is the positive sample, $`I_k^-`$ is the set of negative samples, $`s_i`$ is the model's score for item $`i`$ and $`\sigma`$ is the logistic sigmoid function. 

The $`\beta`$ parameter controls the model calibration level. Note that we do not specify beta directly and infer it from the calibration parameter $`t`$:

```math
\begin{align}
    \beta = \alpha \left(t\left(1 - \frac{1}{\alpha}\right) + \frac{1}{\alpha}\right)
\end{align}
```
Where $`\alpha`$ is the negative sampling rate: $`\frac{`|I_k^-|`}{|I| - 1}`$, and $`|I|`$ is the catalogue size. 


Two models' hyperparameters (in addition to standard SASRec's hyperparameters) are $`k`$ -- the number of negatives per positive, and $`t`$. We recommend using $`k = 256`$ and $`t=0.75`$.  
However, if you want fully calibrated probabilities (e.g., not just to sort items but to use these probabilities as an approximation for CTR), you should set $t=1.0$. In this case, model training will take longer but converge to realistic probabilities (see proofs and experiments in the paper). 

 We do not implement gBCE explicitly. Instead, we use score positive conversion and then use the [standard BCE](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html) loss: 
```math
\begin{align}
        \mathcal{L}^{\beta}_{gBCE}(s^+, s^-) =  \mathcal{L}_{BCE}(\gamma(s^+), s^-)
\end{align}
```
where

```math
\begin{align}
    \gamma(s^+)= \log\left(\frac{1}{\sigma^{-\beta}(s^+) - 1}\right)
\end{align}
```

Corresponding logits transformation code (from model training): 
```python
        alpha = config.negs_per_pos / (num_items - 1)
        t = config.gbce_t 
        beta = alpha * ((1 - 1/alpha)*t + 1/alpha)
        positive_logits = logits[:, :, 0:1].to(torch.float64) #use float64 to increase numerical stability
        negative_logits = logits[:,:,1:].to(torch.float64)
        eps = 1e-10
        positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1-eps)
        positive_probs_adjusted = torch.clamp(positive_probs.pow(-beta), 1+eps, torch.finfo(torch.float64).max)
        to_log = torch.clamp(torch.div(1.0, (positive_probs_adjusted  - 1)), eps, torch.finfo(torch.float64).max)
        positive_logits_transformed = to_log.log()
        logits = torch.cat([positive_logits_transformed, negative_logits], -1)

```

Our gSASRec code is based on the [original SASRec code](https://github.com/kang205/SASRec). The architecture of the model itself is based on a Transformer decoder and is identical to the SASRec architecture. The difference is in training rather than in the model architecture. 

# Config files
We use a Python file as a config file. list of supported parameters: 

**dataset_name**: dataset (from datasets folder).

**sequence_length**: maximum length of the input sequence.

**embedding_dim**: dimensionality of embedding.

**train_batch_size**: gSASRecs batch size used during training.

**num_heads**: number of attention heads used by gSASRec. Note that in SASRec, weights are shared between heads. 

**num_blocks**: number of transformer decoder blocks.

**dropout_rate**: percentage of switched-off neurons. This is important to prevent overfitting on smaller datasets.

**max_epochs**: maximum number of epochs (usually, it will stop earlier).

**negs_per_pos**: gSASRec specific -- number of negatives per positive.

**max_batches_per_epoch**: The epoch number will be limited by this number of batches.

**metrics**: list of metrics used for training or evaluation. 

**val_metric**: This metric will be used for early stopping.

**early_stopping_patience**: Training will stop after this number of epochs without val_metric improving.

**gbce_t**: gSASRec specific, calibration parameter t for the gBCE loss.

**filter_rated**: whether or not filter already rated items

**eval_batch_size**: batch size used during validation

**recommendation_limit**: number of items returned by the `get_predictions` function

**reuse_item_embeddings**: whether or not we re-use original item embeddings for computing output scores. The option to use a separate embeddings table in the output layer is described in the original SASRec paper, but needs to be added to the original code. We added it in this version and used a separate  embedding table by default for gSASRec. 

Default parameter values are specified in the [config.py](config.py) file

Example training configuration for MovieLens-1M: 
```python
from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='ml1m',
    sequence_length=200,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=256,
    gbce_t = 0.75,
)
```
## Metrics
We use [ir-measures](https://ir-measur.es/en/latest/) for measuring model quality. By default, we monitor nDCG@10, Recall@10 (R@10) and Recall@1 (R@1); however, the library contains many more ranking metrics. 

## Datasets
Currently, this dataset only contains a pre-processed version of the MovieLens-1M dataset from the original SASRec repository. The same dataset version was used in the original SASRec paper, BERT4Rec paper, and many other follow-up papers. In this version, all items with less than 5 interactions are removed. 

For reproducibility, we pre-split the dataset into three parts: 

Test: last interaction for each user. 
Validation: second last interactions for 512 randomly selected users. 
Train: all other interactions. 

If you want to map to the original MovieLens-1M dataset (e.g.  for finding item titles or genres), use the reverse-engineered mapping from this repository: https://github.com/asash/ml1m-sas-mapping. 

Following the original dataset, we assume that items are numbered from 1 to num_items, and all sequences are stored as sequences in corresponding folders. 

## SASRec mode

gSASRec is a generalisation over normal SASRec. This code also adapts the original code to pytorch with gSASRec-specific modifications. 

To train a pure SASRec model, set parameters `negs_per_pos=1` (one negative per positive) and `gbce_t=0.0` (this turns gBCE into BCE). The original SASRec  paper suggests that item embeddings can or can not be reused between the first and last layers of the model; however, the original code only has the version with reusable embeddings. In our experiments with gSASRec, we found that a separate set of item embeddings improves model convergence.  Nevertheless, if you want to use this code in vanilla SASRec mode, you can disable separate embeddings. This can be achieved by setting `reuse_item_embeddings=True`. 

Overall, to train a vanilla SASRec model, your config may look like this: 
```python
config = GSASRecExperimentConfig(
    dataset_name='ml1m',
    sequence_length=200,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=1,
    gbce_t = 0.0,
    reuse_item_embeddings=False
)
```
