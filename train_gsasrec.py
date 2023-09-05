from argparse import ArgumentParser
import os

import torch
from utils import load_config, build_model, get_device
from dataset_utils import get_train_dataloader, get_num_items, get_val_dataloader
from tqdm import tqdm
from gbce import gBCE
from eval_utils import evaluate

models_dir = "models"
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_ml1m.py')
args = parser.parse_args()
config = load_config(args.config)

num_items = get_num_items(config.dataset_name) 
device = get_device()
model = build_model(config, device)

train_dataloader = get_train_dataloader(config.dataset_name, batch_size=config.train_batch_size,
                                         max_length=config.sequence_length, train_neg_per_positive=config.negs_per_pos)
val_dataloader = get_val_dataloader(config.dataset_name, batch_size=config.train_batch_size, max_length=config.sequence_length)

loss_fct = gBCE(pool_size=num_items-1, negatives=config.negs_per_pos, t=config.gbce_t)
optimiser = torch.optim.Adam(model.parameters())
batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))

best_metric = float("-inf")
best_model_name = None
step = 0

for epoch in range(config.max_epochs):
    model.train()   
    batch_iter = iter(train_dataloader)
    pbar = tqdm(range(batches_per_epoch))
    loss_sum = 0
    for batch_idx in pbar:
        step += 1
        positives, negatives = [tensor.to(device) for tensor in next(batch_iter)]
        model_input = positives[:, :-1]
        last_hidden_state, attentions = model(model_input)
        labels = positives[:, 1:]
        negatives = negatives[:, 1:, :]
        pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives], dim=-1)
        pos_neg_embeddings = model.item_embedding(pos_neg_concat)
        mask = (labels != num_items + 1).float().unsqueeze(-1).repeat(1, 1, 1, config.negs_per_pos + 1)
        scores = torch.einsum('bld,blnd->bln', last_hidden_state, pos_neg_embeddings)
        gt = torch.zeros_like(scores)
        gt[:, :, 0] = 1
        loss = loss_fct(scores, gt)*mask
        mean_loss = loss.sum() / mask.sum()
        mean_loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        loss_sum += mean_loss.item()
        pbar.set_description(f"Epoch {epoch} loss: {loss_sum / (batch_idx + 1)}")
    evaluation_result = evaluate(model, val_dataloader, config.metrics, config.recommendation_limit, 
                                 config.filter_rated, device=device) 
    print(f"Epoch {epoch} evaluation result: {evaluation_result}")
    if evaluation_result[config.val_metric] > best_metric:
        best_metric = evaluation_result[config.val_metric]
        model_name = f"models/gsasrec-{config.dataset_name}-step:{step}-t:{config.gbce_t}-negs:{config.negs_per_pos}-metric:{best_metric}.pt" 
        print(f"Saving new best model to {model_name}")
        if best_model_name is not None:
            os.remove(best_model_name)
        best_model_name = model_name
        torch.save(model.state_dict(), model_name)


        
