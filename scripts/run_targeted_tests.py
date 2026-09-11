import os
import time
from pprint import pprint

from config import load_config
from utils import ensure_dir, setup_logging, write_json, set_global_seed
from model_factory import create_model
from data_loader import build_data_bundle
from experiment_runner import build_model, train_model
import importlib.util

# load prepare_data_loaders_for_dataset from scripts/run_final_experiments.py
spec = importlib.util.spec_from_file_location("rfe", os.path.join(os.getcwd(), "scripts", "run_final_experiments.py"))
rfe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfe)
prepare_data_loaders_for_dataset = getattr(rfe, "prepare_data_loaders_for_dataset")


RESULTS_ROOT = "results/final_experiments/targeted_tests"
HYBRIDS = ["HybridCNN_GWO_Run1", "HybridCNN_GWO_Run3", "HybridCNN_WOA_Run3"]


def run_mobilenet_cifar(config, epochs=3):
    dataset = "CIFAR10"
    model_name = "mobilenetv3-large"
    batch_size = 16
    seed = config.random_seed
    set_global_seed(seed)
    run_dir = os.path.join(RESULTS_ROOT, dataset, model_name)
    ensure_dir(run_dir)
    logger = setup_logging(os.path.join(run_dir, "run.log"))
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Prepare loaders
    data_dir = getattr(rfe, "DATASETS").get(dataset)
    train_loader, val_loader, test_loader, in_ch, img_size, num_classes = prepare_data_loaders_for_dataset(dataset, data_dir, batch_size, seed, config.num_workers)

    # Build pretrained model
    model = create_model(model_name, num_classes=num_classes, device=device, pretrained=True)
    logger.info("Running pretrained test: %s %s bs=%s device=%s", dataset, model_name, batch_size, device)

    try:
        model, history, best_val_acc, best_epoch, train_time = train_model(
            model,
            train_loader,
            val_loader,
            device,
            learning_rate=config.learning_rate,
            epochs=epochs,
            patience=config.patience,
            logger=logger,
            checkpoint_path=os.path.join(run_dir, "best_model.pth"),
        )
        result = {
            "status": "SUCCESS",
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "training_time_seconds": train_time,
        }
    except Exception as e:
        logger.exception("Pretrained mobilenet test failed: %s", e)
        result = {"status": "FAILED", "exception": str(e)}

    write_json(os.path.join(run_dir, "result.json"), result)
    return result


def run_hybrid_trials(config, datasets=("ISIC2019", "BrainTumor"), epochs=3):
    seed = config.random_seed
    set_global_seed(seed)
    results = {}
    for dataset in datasets:
        for model_name in HYBRIDS:
            run_dir = os.path.join(RESULTS_ROOT, dataset, model_name)
            ensure_dir(run_dir)
            logger = setup_logging(os.path.join(run_dir, "run.log"))
            logger.info("Hybrid trial start: %s %s", dataset, model_name)
            # Load summary for hybrid to obtain hyperparams
            mapping = {
                "HybridCNN_GWO_Run1": "results/CIFAR10/GWO/run_01/summary.json",
                "HybridCNN_GWO_Run3": "results/CIFAR10/GWO/run_03/summary.json",
                "HybridCNN_WOA_Run3": "results/CIFAR10/WOA/run_03/summary.json",
            }
            summary_path = mapping.get(model_name)
            if not summary_path or not os.path.exists(summary_path):
                logger.warning("Missing hybrid summary: %s", summary_path)
                write_json(os.path.join(run_dir, "result.json"), {"status": "MISSING_SUMMARY"})
                results[(dataset, model_name)] = {"status": "MISSING_SUMMARY"}
                continue
            summary = __import__("utils").read_json(summary_path)
            best_hp = summary.get("best_hyperparameters", {})
            # ensure num_classes from dataset
            # prepare trials for batch sizes
            trial_bss = [16, 12, 8]
            success = False
            for bs in trial_bss:
                logger.info("Trying batch_size=%s for %s on %s", bs, model_name, dataset)
                try:
                    data_dir = getattr(rfe, "DATASETS").get(dataset)
                    train_loader, val_loader, test_loader, in_ch, img_size, num_classes = prepare_data_loaders_for_dataset(dataset, data_dir, bs, seed, config.num_workers)
                    # build model from best_hp
                    model_cfg = best_hp.copy()
                    model = build_model(model_cfg, in_ch, img_size, num_classes)
                    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
                    model = model.to(device)
                    model, history, best_val_acc, best_epoch, train_time = train_model(
                        model,
                        train_loader,
                        val_loader,
                        device,
                        learning_rate=model_cfg.get("learning_rate", config.learning_rate),
                        epochs=epochs,
                        patience=config.patience,
                        logger=logger,
                        checkpoint_path=os.path.join(run_dir, "best_model.pth"),
                    )
                    # success
                    res = {"status": "SUCCESS", "batch_size": bs, "best_val_accuracy": best_val_acc, "best_epoch": best_epoch, "training_time_seconds": train_time}
                    write_json(os.path.join(run_dir, "result.json"), res)
                    results[(dataset, model_name)] = res
                    success = True
                    break
                except Exception as e:
                    # check if OOM
                    msg = str(e)
                    logger.exception("Trial bs=%s failed: %s", bs, e)
                    if "out of memory" in msg.lower() or "cuda out of memory" in msg.lower():
                        logger.info("OOM at bs=%s, will try smaller batch", bs)
                        continue
                    else:
                        write_json(os.path.join(run_dir, "result.json"), {"status": "FAILED", "exception": msg})
                        results[(dataset, model_name)] = {"status": "FAILED", "exception": msg}
                        success = False
                        break
            if not success:
                # if not already recorded (e.g., all OOMs)
                if (dataset, model_name) not in results or results[(dataset, model_name)]["status"] != "SUCCESS":
                    write_json(os.path.join(run_dir, "result.json"), {"status": "FAILED_OR_OOMS"})
                    results[(dataset, model_name)] = {"status": "FAILED_OR_OOMS"}
    return results


if __name__ == "__main__":
    cfg = load_config(None)
    ensure_dir(RESULTS_ROOT)
    print("Running targeted tests: pretrained baselines + hybrid batch-size trials")
    t0 = time.time()

    # Baseline pretrained sweep across datasets
    baselines = ["efficientnet-b0", "mobilenetv3-large", "shufflenetv2-1.5x"]
    all_results = {}
    for ds in ["CIFAR10", "ISIC2019", "BrainTumor"]:
        for m in baselines:
            run_dir = os.path.join(RESULTS_ROOT, ds, m)
            ensure_dir(run_dir)
            logger = setup_logging(os.path.join(run_dir, "run.log"))
            logger.info("Pretrained baseline test start: dataset=%s model=%s", ds, m)
            try:
                data_dir = getattr(rfe, "DATASETS").get(ds)
                train_loader, val_loader, test_loader, in_ch, img_size, num_classes = prepare_data_loaders_for_dataset(ds, data_dir, cfg.batch_size, cfg.random_seed, cfg.num_workers)
                device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
                model = create_model(m, num_classes=num_classes, device=device, pretrained=True)
                model, history, best_val_acc, best_epoch, train_time = train_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    learning_rate=cfg.learning_rate,
                    epochs=3,
                    patience=cfg.patience,
                    logger=logger,
                    checkpoint_path=os.path.join(run_dir, "best_model.pth"),
                )
                res = {"status": "SUCCESS", "best_val_accuracy": best_val_acc, "best_epoch": best_epoch, "training_time_seconds": train_time}
            except Exception as e:
                logger.exception("Pretrained baseline test failed: %s", e)
                res = {"status": "FAILED", "exception": str(e)}
            write_json(os.path.join(run_dir, "result.json"), res)
            all_results[(ds, m)] = res

    pprint(("pretrained_baseline_results", all_results))

    # Hybrid trials
    r2 = run_hybrid_trials(cfg, datasets=("ISIC2019", "BrainTumor"), epochs=3)
    pprint(("hybrid_results", r2))
    print("Done in", time.time() - t0)
