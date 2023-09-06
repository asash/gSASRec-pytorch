This is an official pytorch version of the gSASRec model.


This model has been validated against the original gSASRec tensorflow implementation.
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
