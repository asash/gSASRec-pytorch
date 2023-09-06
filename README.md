# gSASRec-pytorch

This is an official port of the gSASRec model to pytorch. The model is described in the RecSys '23 paper **"gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling"** by Aleksandr Petrov and Craig Macdonald (University of Glasgow). 

**Link to the paper:**[https://arxiv.org/pdf/2308.07192.pdf](https://arxiv.org/pdf/2308.07192.pdf)

If you use this code from the repository, please cite the work: 
```
@inproceedings{petrov2022recencysampling,
  title={gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling},
  author={Petrov, Aleksandr and Macdonald, Craig},
  booktitle={Seventeen ACM Conference on Recommender Systems},
  year={2022}
}
```


This model has been validated against the original gSASRec tensorflow implementation.
This repository also contains a pre-split MovieLens-1M dataset and a pre-trained checkpoint of gSASRec for reproducibility. 


## GSASrec and GBCE info info
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
However, if you want fully calibrated probabilities (e.g., not just to sort items but use these probabilities as an approximation for CTR), you should set $t=1.0$. In this case, model training will take longer but converge to realistic probabilities (see proofs and experiments in the paper). 

 We do not implement gBCE explicitly. Instead, we use score positive conversion and then use the [standard BCE](losses/bce.py) loss: 
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

Our SASRec code is based on the original SASRec code. 

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


## SASRec mode

gSASRec is a generalisation over normal SASRec. This code also adapts the original code to pytorch with gSASRec-specific modifications. 

To train a pure SASRec model, set parameters `negs_per_pos=1` (one negative per positive) and `gbce_t=0.0` (use gBCE in the original form). The original SASRec  paper suggests that item embeddings can or can not be reused between the first and last layers of the model; however, the original code only has the version with reusable embeddings. In our experiments with gSASRec, we found that a separate set of item embeddings improves model convergence.  Nevertheless, if you want to use this code in vanilla SASRec mode, you can disable separate embeddings. This can be achieved by setting `reuse_item_embeddings=True`. 

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
