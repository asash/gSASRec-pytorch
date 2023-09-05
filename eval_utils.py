import torch
import tqdm 
from gsasrec import GSASRec
from ir_measures import ScoredDoc, Qrel
import ir_measures

def evaluate(model: GSASRec, data_loader, metrics, limit, filter_rated, device):
    model.eval()
    users_processed = 0
    scored_docs = []
    qrels = [] 
    with torch.no_grad():
        max_batches = len(data_loader)
        for batch_idx, (data, rated, target) in tqdm.tqdm(enumerate(data_loader), total=max_batches):
            data, target = data.to(device), target.to(device)
            if filter_rated:
                items, scores = model.get_predictions(data, limit, rated)
            else:
                items, scores = model.get_predictions(data, limit)
            for recommended_items, recommended_scores, target in zip(items, scores, target):
                for item, score in zip(recommended_items, recommended_scores):
                    scored_docs.append(ScoredDoc(str(users_processed), str(item.item()), score.item()))
                qrels.append(Qrel(str(users_processed), str(target.item()), 1))
                users_processed += 1
                pass
    result = ir_measures.calc_aggregate(metrics, qrels, scored_docs)
    return result
    