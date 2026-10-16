import importlib.util, os, sys

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("rfe", "scripts/run_final_experiments.py")
rfe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfe)
from config import load_config

cfg = load_config(None)
cfg.final_epochs = 1
cfg.batch_size = 4
cfg.num_workers = 0
print("Running: ISIC2019 HybridCNN_GWO_Run1, batch_size=", cfg.batch_size)
res = rfe.run_single_experiment(
    "ISIC2019",
    "HybridCNN_GWO_Run1",
    cfg,
    epochs=1,
    baseline_lr=cfg.learning_rate,
    baseline_batch=cfg.batch_size,
    start_flag=True,
)
print("Result:", res)
