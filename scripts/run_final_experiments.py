import os
import sys
import time
import json
import csv
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from utils import ensure_dir, set_global_seed, write_json, read_json, setup_logging
from data_loader import build_data_bundle
from model_factory import create_model
from experiment_runner import build_model, train_model, summarize_confusion_matrix
from metrics import evaluate_model, build_confusion_matrix

# Configuration for final experiments
RESULTS_ROOT = "results/final_experiments"
DATASETS_ROOT = r"C:\Users\emirh\Desktop\Projects\datasets"

# Dataset paths (will validate existence)
DATASETS = {
    "CIFAR10": None,  # use internal loader
    "ISIC2019": os.path.join(DATASETS_ROOT, "input_sk"),
    "BrainTumor": os.path.join(DATASETS_ROOT, "BT"),
}

MODELS = [
    "HybridCNN_GWO_Run1",
    "HybridCNN_GWO_Run3",
    "HybridCNN_WOA_Run3",
    "efficientnet-b0",
    "mobilenetv3-large",
    "shufflenetv2-1.5x",
]

# When True, force all experiments to use the same training settings (batch, lr) from config
USE_UNIFORM_SETTINGS = True

MASTER_CSV = os.path.join(RESULTS_ROOT, "master_results.csv")
MASTER_JSON = os.path.join(RESULTS_ROOT, "master_results.json")
FINAL_SUMMARY = os.path.join(RESULTS_ROOT, "final_summary.json")

# Smoke-test flags
SMOKE_ONLY = True


def ensure_dataset_paths():
    missing = []
    for name, path in DATASETS.items():
        if path is None:
            continue
        if not os.path.exists(path):
            missing.append((name, path))
    return missing


def check_brain_tumor_structure(bt_path):
    # Expect TRAIN and TEST directories
    train = os.path.join(bt_path, "TRAIN")
    test = os.path.join(bt_path, "TEST")
    return os.path.isdir(train), os.path.isdir(test)


def infer_num_classes_from_folder(path):
    # assume structure: path/<split>/<class> or path/<class> for test dirs
    for root, dirs, files in os.walk(path):
        # look for class directories directly under root
        if dirs:
            return len(dirs)
    return None


def import_existing_cifar_hybrid_results(target_dir):
    # copy existing JSON results for the 3 hybrid runs into target_dir/CIFAR10/Hybrid...
    # Look for legacy summaries in either results/CIFAR10 or results/main_experiment
    candidates = [os.path.join("results", "CIFAR10"), os.path.join("results", "main_experiment")]
    hybrids = [
        ("HybridCNN_GWO_Run1", os.path.join("GWO", "run_01", "summary.json")),
        ("HybridCNN_GWO_Run3", os.path.join("GWO", "run_03", "summary.json")),
        ("HybridCNN_WOA_Run3", os.path.join("WOA", "run_03", "summary.json")),
    ]
    imported = []
    for name, relpath in hybrids:
        # search candidate bases for the relative path
        found = False
        for base in candidates:
            src = os.path.join(base, relpath)
            if os.path.exists(src):
                dst_dir = os.path.join(target_dir, "CIFAR10", name)
                ensure_dir(dst_dir)
                try:
                    data = read_json(src)
                    write_json(os.path.join(dst_dir, "result.json"), data)
                    imported.append(name)
                    found = True
                    break
                except Exception:
                    pass
        if not found:
            # no-op if missing
            continue
    return imported


def smoke_test_dataset_and_models(config):
    summary = {"datasets": {}, "models": {}}
    set_global_seed(config.random_seed)

    # datasets
    missing = ensure_dataset_paths()
    for name, path in DATASETS.items():
        ds_info = {"path": path, "exists": True, "details": {}}
        if path is None:
            ds_info["exists"] = True
            ds_info["details"]["note"] = "CIFAR10 uses internal loader"
        else:
            if not os.path.exists(path):
                ds_info["exists"] = False
            else:
                if name == "BrainTumor":
                    train_ok, test_ok = check_brain_tumor_structure(path)
                    ds_info["details"]["train_dir_exists"] = train_ok
                    ds_info["details"]["test_dir_exists"] = test_ok
                    if train_ok:
                        ds_info["details"]["train_num_classes"] = infer_num_classes_from_folder(os.path.join(path, "TRAIN"))
                    if test_ok:
                        ds_info["details"]["test_num_classes"] = infer_num_classes_from_folder(os.path.join(path, "TEST"))
                elif name == "ISIC2019":
                    ds_info["details"]["num_classes"] = infer_num_classes_from_folder(path)
        summary["datasets"][name] = ds_info

    # models: create and dummy forward with sample input sizes per dataset
    for model_name in MODELS:
        try:
            m_cpu = create_model(model_name, num_classes=10, device="cpu")
            # dummy forward for CIFAR10 shape
            import torch

            x = torch.randn(2, 3, 32, 32)
            y = m_cpu(x)
            out_shape = list(y.shape)
            summary["models"][model_name] = {"created": True, "dummy_output_shape": out_shape}
        except Exception as e:
            summary["models"][model_name] = {"created": False, "error": str(e)}

    return summary


def write_master_files(master_rows, master_json):
    ensure_dir(RESULTS_ROOT)
    # write csv
    with open(MASTER_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset",
            "model",
            "seed",
            "total_parameters",
            "trainable_parameters",
            "best_epoch",
            "best_val_accuracy",
            "test_accuracy",
            "test_precision_macro",
            "test_recall_macro",
            "test_f1_macro",
            "validation_accuracy",
            "validation_precision_macro",
            "validation_recall_macro",
            "validation_f1_macro",
            "training_time_seconds",
            "total_wall_time_seconds",
            "best_model_path",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in master_rows:
            writer.writerow(r)

    # write json
    write_json(master_json, {"rows": master_rows})


def prepare_data_loaders_for_dataset(dataset, data_dir, batch_size, seed, num_workers):
    """Return train_loader, val_loader, test_loader, in_channels, img_size, num_classes
    For CIFAR10 uses build_data_bundle; for ISIC2019 and BrainTumor uses ImageFolder conventions."""
    import torch
    from torch.utils.data import random_split, DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision import transforms

    if dataset.lower() == "cifar10":
        bundle = build_data_bundle(
            dataset=dataset,
            data_dir=data_dir or "data",
            batch_size=batch_size,
            train_size=45000,
            val_size=5000,
            seed=seed,
            num_workers=num_workers,
        )
        return bundle.train_loader, bundle.val_loader, bundle.test_loader, bundle.in_channels, bundle.img_size, bundle.num_classes

    # Common transforms for ImageFolder datasets
    # Ensure images are converted to RGB so models expecting 3 channels won't break on grayscale data
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if dataset.lower() == "isic2019":
        # Prefer train/val/test dirs
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        test_dir = os.path.join(data_dir, "test")
        if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
            train_ds = ImageFolder(train_dir, transform=transform)
            val_ds = ImageFolder(val_dir, transform=transform)
            test_ds = ImageFolder(test_dir, transform=transform)
        elif os.path.isdir(train_dir) and os.path.isdir(test_dir):
            train_ds = ImageFolder(train_dir, transform=transform)
            # split a small val from train (10%)
            total = len(train_ds)
            val_n = max(1, int(total * 0.1))
            train_n = total - val_n
            g = torch.Generator().manual_seed(seed)
            train_ds, val_ds = random_split(train_ds, [train_n, val_n], generator=g)
            test_ds = ImageFolder(test_dir, transform=transform)
        else:
            # fallback: single folder with classes
            ds = ImageFolder(data_dir, transform=transform)
            total = len(ds)
            # create train/val/test split 80/10/10
            n_train = int(total * 0.8)
            n_val = int(total * 0.1)
            n_test = total - n_train - n_val
            g = torch.Generator().manual_seed(seed)
            train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=g)

        def make_loader(ds, shuffle):
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        in_channels = 3
        img_size = 224
        # num_classes: if ImageFolder use classes, else infer from dataset
        num_classes = None
        try:
            if isinstance(train_ds, ImageFolder):
                num_classes = len(train_ds.classes)
            else:
                # random_split returns Subset
                num_classes = len(train_ds.dataset.classes)
        except Exception:
            num_classes = 0

        return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False), in_channels, img_size, num_classes

    if dataset.lower() in ("braintumor", "brain-tumor", "brain_tumor"):
        train_root = os.path.join(data_dir, "TRAIN")
        test_root = os.path.join(data_dir, "TEST")
        if not os.path.isdir(train_root) or not os.path.isdir(test_root):
            raise ValueError("BrainTumor dataset must contain TRAIN and TEST directories")
        train_ds = ImageFolder(train_root, transform=transform)
        val_ds = ImageFolder(test_root, transform=transform)
        # No separate test set
        test_ds = None

        def make_loader(ds, shuffle):
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        in_channels = 3
        img_size = 224
        num_classes = len(train_ds.classes)
        return make_loader(train_ds, True), make_loader(val_ds, False), None, in_channels, img_size, num_classes


def run_single_experiment(dataset, model_name, config, epochs, baseline_lr, baseline_batch, start_flag=False):
    """Run one dataset x model experiment. If start_flag is False, perform smoke actions only (dummy forward, parameter count)."""
    seed = config.random_seed
    set_global_seed(seed)

    run_dir = os.path.join(RESULTS_ROOT, dataset, model_name)
    ensure_dir(run_dir)
    run_log = os.path.join(run_dir, "run.log")
    logger = setup_logging(run_log)

    # Determine hyperparams and create model
    is_hybrid = model_name.lower().startswith("hybridcnn")

    # For HybridCNN use build_model with hyperparameters from summary
    if is_hybrid:
        # map names to relative summary paths and search candidate bases
        mapping = {
            "hybridcnn_gwo_run1": os.path.join("GWO", "run_01", "summary.json"),
            "hybridcnn_gwo_run3": os.path.join("GWO", "run_03", "summary.json"),
            "hybridcnn_woa_run3": os.path.join("WOA", "run_03", "summary.json"),
        }
        rel = mapping.get(model_name.lower())
        candidates = [os.path.join("results", "CIFAR10"), os.path.join("results", "main_experiment")]
        summary = None
        if rel:
            for base in candidates:
                p = os.path.join(base, rel)
                if os.path.exists(p):
                    try:
                        summary = read_json(p)
                        break
                    except Exception:
                        continue
        if summary is None:
            return {"status": "FAILED", "reason": "missing hybrid summary"}
        best_hp = summary.get("best_hyperparameters", {})
        # model_config uses architecture hyperparams from the search summary
        model_config = best_hp
        # training hyperparams: either force uniform from config, or use search-found values
        if USE_UNIFORM_SETTINGS:
            lr = config.learning_rate
            batch_size = config.batch_size
        else:
            lr = best_hp.get("learning_rate", config.learning_rate)
            batch_size = best_hp.get("batch_size", config.batch_size)
    else:
        # baseline
        model_config = {"model_name": model_name}
        if USE_UNIFORM_SETTINGS:
            lr = config.learning_rate
            batch_size = config.batch_size
        else:
            lr = baseline_lr
            batch_size = baseline_batch

    logger.info(
        "Starting experiment: dataset=%s model=%s hybrid=%s lr=%s batch_size=%s seed=%s",
        dataset,
        model_name,
        is_hybrid,
        lr,
        batch_size,
        seed,
    )

    # Prepare data loaders
    data_dir = DATASETS.get(dataset)

    # For hybrid models we may need to retry with smaller batch sizes on OOM
    tried_batch = None
    if is_hybrid:
        # batch candidate list: prefer best found batch_size, then common fallbacks
        hp_bs = batch_size
        candidates = []
        if isinstance(hp_bs, int) and hp_bs > 0:
            candidates.append(hp_bs)
        for b in (16, 12, 8):
            if b not in candidates:
                candidates.append(b)
    else:
        candidates = [batch_size]

    # Try training with candidate batch sizes; if OOM occurs during loader build or training, retry with smaller bs
    last_exception = None
    successful = False
    for bs_try in candidates:
        tried_batch = bs_try
        try:
            train_loader, val_loader, test_loader, in_channels, img_size, num_classes = prepare_data_loaders_for_dataset(dataset, data_dir, bs_try, seed, config.num_workers)

            # Build model for this attempt
            if is_hybrid:
                model = build_model(model_config, in_channels, img_size, num_classes)
            else:
                model = create_model(model_name, num_classes=num_classes, device="cpu")

            # Parameter counts
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("Attempt bs=%s: Model built: total_params=%d trainable_params=%d", bs_try, int(total), int(trainable))

            # If not start_flag, do smoke: dummy forward and return
            if not start_flag:
                write_json(os.path.join(run_dir, "result.json"), {"status": "SKIPPED_SMOKE"})
                return {"status": "SKIPPED_SMOKE"}

            # TRAINING
            try:
                device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
                model = model.to(device)
                start_time = time.perf_counter()
                model, history, best_val_acc, best_epoch, train_time = train_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    learning_rate=lr,
                    epochs=epochs,
                    patience=config.patience,
                    logger=logger,
                    checkpoint_path=os.path.join(run_dir, "best_model.pth"),
                )
                total_time = time.perf_counter() - start_time
                successful = True
                batch_size = bs_try
                break
            except Exception as e:
                last_exception = e
                msg = str(e).lower()
                logger.exception("Training attempt bs=%s failed: %s", bs_try, e)
                if "out of memory" in msg or "cuda out of memory" in msg:
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    logger.info("OOM at bs=%s, will try smaller batch", bs_try)
                    continue
                else:
                    write_json(os.path.join(run_dir, "result.json"), {"status": "FAILED", "exception": str(e)})
                    return {"status": "FAILED", "exception": str(e)}

        except Exception as e:
            last_exception = e
            msg = str(e).lower()
            logger.exception("Preparation attempt bs=%s failed: %s", bs_try, e)
            if "out of memory" in msg or "cuda out of memory" in msg:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                logger.info("OOM while preparing loaders at bs=%s, will try smaller batch", bs_try)
                continue
            else:
                write_json(os.path.join(run_dir, "result.json"), {"status": "FAILED", "exception": str(e)})
                return {"status": "FAILED", "exception": str(e)}

    if not successful:
        msg = str(last_exception) if last_exception else "training failed for unknown reasons"
        logger.exception("All attempts failed for %s/%s: %s", dataset, model_name, msg)
        write_json(os.path.join(run_dir, "result.json"), {"status": "FAILED", "exception": msg})
        return {"status": "FAILED", "exception": msg}

    # Evaluation
    if test_loader is not None:
        test_metrics = evaluate_model(model, test_loader, device)
        confusion = build_confusion_matrix(test_metrics["targets"], test_metrics["predictions"], num_classes)
        summary_scores = summarize_confusion_matrix(confusion)
    else:
        test_metrics = None
        confusion = None
        summary_scores = {}

    # Log training summary
    logger.info(
        "Finished training: dataset=%s model=%s best_epoch=%s best_val_acc=%.6f training_time=%.1fs total_time=%.1fs",
        dataset,
        model_name,
        best_epoch,
        best_val_acc,
        train_time,
        total_time,
    )

    if test_metrics is not None:
        try:
            logger.info(
                "Test results: accuracy=%.4f precision_macro=%.4f recall_macro=%.4f f1_macro=%.4f",
                test_metrics.get("accuracy", 0.0),
                test_metrics.get("precision_macro", 0.0),
                test_metrics.get("recall_macro", 0.0),
                test_metrics.get("f1_macro", 0.0),
            )
        except Exception:
            pass

    result = {
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "parameter_count": {"total": int(total), "trainable": int(trainable)},
        "history": history,
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "summary_scores": summary_scores,
        "training_time_seconds": train_time,
        "total_time_seconds": total_time,
        "best_model_path": os.path.join(run_dir, "best_model.pth"),
        "status": "SUCCESS",
    }
    write_json(os.path.join(run_dir, "result.json"), result)
    if confusion is not None:
        write_json(os.path.join(run_dir, "confusion_matrix.json"), {"matrix": confusion.tolist()})
    write_json(os.path.join(run_dir, "training_history.json"), history)
    return {"status": "SUCCESS"}


def main():
    # Run full experiments by default when executed (no CLI arguments required).
    config = load_config(None)
    ensure_dir(RESULTS_ROOT)

    # Import existing CIFAR hybrids into master structure
    imported = import_existing_cifar_hybrid_results(RESULTS_ROOT)
    print("Imported existing CIFAR HybridCNN results:", imported)

    # baseline lr/batch from HybridCNN_GWO_Run1
    gwo1_path = os.path.join("results", "CIFAR10", "GWO", "run_01", "summary.json")
    baseline_lr = config.learning_rate
    baseline_batch = config.batch_size
    if os.path.exists(gwo1_path):
        try:
            s = read_json(gwo1_path)
            bh = s.get("best_hyperparameters", {})
            baseline_lr = bh.get("learning_rate", baseline_lr)
            baseline_batch = bh.get("batch_size", baseline_batch)
        except Exception:
            pass

    # Build master rows and run experiments (start=True)
    master_rows = []
    total_start = time.perf_counter()
    epochs = getattr(config, "final_epochs", 50)
    for dataset in ["CIFAR10", "ISIC2019", "BrainTumor"]:
        for model in MODELS:
            # Resume logic: skip if result.json and best_model.pth exist
            run_dir = os.path.join(RESULTS_ROOT, dataset, model)
            ensure_dir(run_dir)
            result_json_path = os.path.join(run_dir, "result.json")
            best_model_path = os.path.join(run_dir, "best_model.pth")

            # If this is CIFAR10 and a HybridCNN, prefer importing existing summary and skip training
            if dataset == "CIFAR10" and model.lower().startswith("hybridcnn"):
                # try to import existing CIFAR hybrid summary
                mapping = {
                    "hybridcnn_gwo_run1": os.path.join("results", "CIFAR10", "GWO", "run_01", "summary.json"),
                    "hybridcnn_gwo_run3": os.path.join("results", "CIFAR10", "GWO", "run_03", "summary.json"),
                    "hybridcnn_woa_run3": os.path.join("results", "CIFAR10", "WOA", "run_03", "summary.json"),
                }
                summary_src = mapping.get(model.lower())
                if summary_src and os.path.exists(summary_src):
                    try:
                        data = read_json(summary_src)
                        write_json(result_json_path, data)
                        status = "IMPORTED"
                        print(f"Imported existing hybrid summary for {dataset}/{model}")
                    except Exception:
                        status = "FAILED"
                else:
                    # No existing summary to import; fall back to normal behavior
                    if os.path.exists(result_json_path) and os.path.exists(best_model_path):
                        status = "SKIPPED_ALREADY"
                        print(f"Skipping existing: {dataset}/{model}")
                    else:
                        info = run_single_experiment(dataset, model, config, epochs, baseline_lr, baseline_batch, start_flag=True)
                        status = info.get("status", "FAILED")
            else:
                if os.path.exists(result_json_path) and os.path.exists(best_model_path):
                    status = "SKIPPED_ALREADY"
                    print(f"Skipping existing: {dataset}/{model}")
                else:
                    info = run_single_experiment(dataset, model, config, epochs, baseline_lr, baseline_batch, start_flag=True)
                    status = info.get("status", "FAILED")

            # minimal row for master
            row = {
                "dataset": dataset,
                "model": model,
                "seed": config.random_seed,
                "total_parameters": None,
                "trainable_parameters": None,
                "best_epoch": None,
                "best_val_accuracy": None,
                "test_accuracy": None,
                "test_precision_macro": None,
                "test_recall_macro": None,
                "test_f1_macro": None,
                "validation_accuracy": None,
                "validation_precision_macro": None,
                "validation_recall_macro": None,
                "validation_f1_macro": None,
                "training_time_seconds": None,
                "total_wall_time_seconds": None,
                "best_model_path": best_model_path if os.path.exists(best_model_path) else None,
                "status": status,
            }
            # if result.json exists, fill fields
            if os.path.exists(result_json_path):
                try:
                    r = read_json(result_json_path)
                    row["best_epoch"] = r.get("best_epoch")
                    row["best_val_accuracy"] = r.get("best_val_accuracy")
                    tm = r.get("test_metrics")
                    if tm:
                        row["test_accuracy"] = tm.get("accuracy")
                    pc = r.get("parameter_count")
                    if pc:
                        row["total_parameters"] = pc.get("total")
                        row["trainable_parameters"] = pc.get("trainable")
                    row["training_time_seconds"] = r.get("training_time_seconds")
                except Exception:
                    pass

            master_rows.append(row)

    total_time = time.perf_counter() - total_start
    write_master_files(master_rows, MASTER_JSON)
    final_summary = {
        "total_experiments": len(master_rows),
        "successful_experiments": sum(1 for r in master_rows if r["status"] == "SUCCESS"),
        "failed_experiments": sum(1 for r in master_rows if r["status"] == "FAILED"),
        "skipped_experiments": sum(1 for r in master_rows if r["status"].startswith("SKIPPED")),
        "total_runtime": total_time,
        "dataset_status": {k: (v is not None) for k, v in DATASETS.items()},
        "model_status": {m: True for m in MODELS},
        "master_results_csv": MASTER_CSV,
        "master_results_json": MASTER_JSON,
    }
    write_json(FINAL_SUMMARY, final_summary)
    print("Orchestration complete. Master files:", MASTER_CSV, MASTER_JSON)


if __name__ == "__main__":
    main()
