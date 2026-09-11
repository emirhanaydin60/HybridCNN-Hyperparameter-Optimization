import os
import time
from pprint import pprint
import importlib.util

# load orchestrator module
spec = importlib.util.spec_from_file_location("rfe", os.path.join(os.getcwd(), "scripts", "run_final_experiments.py"))
rfe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfe)

from config import load_config
from utils import ensure_dir, setup_logging, write_json
from model_factory import create_model
from experiment_runner import train_model

RESULTS_ROOT = "results/final_experiments/quick_checks"
BASELINES = ["efficientnet-b0", "mobilenetv3-large", "shufflenetv2-1.5x"]
HYBRID = "HybridCNN_GWO_Run1"


def run_isic_pretrained(config):
    dataset = "ISIC2019"
    data_dir = rfe.DATASETS.get(dataset)
    results = {}
    for m in BASELINES:
        run_dir = os.path.join(RESULTS_ROOT, dataset, m)
        ensure_dir(run_dir)
        logger = setup_logging(os.path.join(run_dir, "run.log"))
        logger.info("Quick pretrained baseline start: %s %s", dataset, m)
        try:
            train_loader, val_loader, test_loader, in_ch, img_size, num_classes = rfe.prepare_data_loaders_for_dataset(dataset, data_dir, config.batch_size, config.random_seed, config.num_workers)
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            model = create_model(m, num_classes=num_classes, device=device, pretrained=True)
            model, history, best_val_acc, best_epoch, train_time = train_model(
                model,
                train_loader,
                val_loader,
                device,
                learning_rate=config.learning_rate,
                epochs=3,
                patience=config.patience,
                logger=logger,
                checkpoint_path=os.path.join(run_dir, "best_model.pth"),
            )
            res = {"status": "SUCCESS", "best_val_accuracy": best_val_acc, "best_epoch": best_epoch, "training_time_seconds": train_time}
        except Exception as e:
            logger.exception("Pretrained baseline failed: %s", e)
            res = {"status": "FAILED", "exception": str(e)}
        write_json(os.path.join(run_dir, "result.json"), res)
        results[m] = res
    return results


def run_hybrid_check(config, dataset, hybrid_name=HYBRID):
    run_dir = os.path.join(RESULTS_ROOT, dataset, hybrid_name)
    ensure_dir(run_dir)
    logger = setup_logging(os.path.join(run_dir, "run.log"))
    logger.info("Quick hybrid smoke start: %s %s", dataset, hybrid_name)
    try:
        info = rfe.run_single_experiment(dataset, hybrid_name, config, epochs=1, baseline_lr=config.learning_rate, baseline_batch=config.batch_size, start_flag=True)
    except Exception as e:
        logger.exception("Hybrid quick check failed: %s", e)
        info = {"status": "FAILED", "exception": str(e)}
    return info


if __name__ == "__main__":
    cfg = load_config(None)
    ensure_dir(RESULTS_ROOT)
    t0 = time.time()
    print("Running ISIC2019 pretrained baselines (3 epochs) and 1-epoch hybrid smoke tests...")
    bas_res = run_isic_pretrained(cfg)
    pprint(("isic_pretrained_results", bas_res))
    hybrid_isic = run_hybrid_check(cfg, "ISIC2019", HYBRID)
    hybrid_bt = run_hybrid_check(cfg, "BrainTumor", HYBRID)
    pprint(("hybrid_isic", hybrid_isic))
    pprint(("hybrid_bt", hybrid_bt))
    print("Done in", time.time() - t0)
