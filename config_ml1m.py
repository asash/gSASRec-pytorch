from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='ml1m',
    sequence_length=200,
    embedding_dim=256,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=3,
    dropout_rate=0.1,
    negs_per_pos=1,
    gbce_t = 0.0,
)
