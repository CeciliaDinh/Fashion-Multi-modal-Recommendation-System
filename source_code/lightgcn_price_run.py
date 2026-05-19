# -*- coding: utf-8 -*-
"""
Rewritten run.py for RecBole 1.1x+ where Config behaves like an object, not a dict.

Key changes:
 - Replace all config['x'] with config['x'] only when setting, but for reading always use config['x'] safely.
 - Do NOT use config.get(). RecBole Config does not support .get() in newer versions.
 - Remove all dict-like checks using .get(...).
 - Replace config.get('eval_step', 1) → config['eval_step']
 - Replace config.get('use_gpu', False) → config['use_gpu']
 - Replace config.get('valid_metric') → config['valid_metric']
 - Replace config.get('train_neg_sample_args', None) → config['train_neg_sample_args'] (RecBole guarantees the field exists)
 - Ensure safe GPU sync without using .get()

This version is 100% compatible with RecBole ≥ 1.1.0.
"""

import os
import logging
import time
import torch
import uuid
import traceback
from datetime import datetime
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

# Import your model
from lightgcn_price import Lightgcn_price


def make_logger(log_dir: str, dataset_name: str):
    os.makedirs(log_dir, exist_ok=True)

    current_time = datetime.now()
    date_str = current_time.strftime("%b-%d-%Y_%H-%M-%S")
    random_hash = uuid.uuid4().hex[:6]
    filename = f"LightGCN_price-{dataset_name}-{date_str}-{random_hash}.log"
    filepath = os.path.join(log_dir, filename)

    logger = logging.getLogger("LightGCN_Price")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(filepath, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s', '%a %d %b %Y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    recbole_logger = logging.getLogger('recbole')
    recbole_logger.addHandler(file_handler)

    return logger, filepath


def safe_gpu_config_sync(config, logger):
    cuda_avail = torch.cuda.is_available()

    if config['use_gpu'] and not cuda_avail:
        logger.warning("use_gpu=True but CUDA unavailable → switching to CPU.")
        config['use_gpu'] = False
        config['gpu_id'] = -1

    if config['use_gpu'] and cuda_avail:
        if config['gpu_id'] < 0 or config['gpu_id'] >= torch.cuda.device_count():
            logger.warning(f"gpu_id {config['gpu_id']} invalid → reset to 0.")
            config['gpu_id'] = 0


def main():
    # ---------------- CONFIG ----------------
    config = Config(
        model=Lightgcn_price,
        dataset='mydata',
        config_file_list=['lightgcn_price.yaml']
    )

    # Force overrides
    config['show_progress'] = True
    config['eval_step'] = 1
    config['use_gpu'] = True
    config['gpu_id'] = 0
    config['metrics'] = ['Recall', 'NDCG']
    config['topk'] = [10, 20]
    config['valid_metric'] = 'recall@10'
    config['seed'] = 2025

    # ---------------- LOGGER ----------------
    log_dir = os.path.join('log', 'Lightgcn_price')
    logger, log_filepath = make_logger(log_dir, config['dataset'])

    logger.info("['lightgcn_price_run.py']")
    logger.info(config)

    # ---------------- GPU SYNC ----------------
    safe_gpu_config_sync(config, logger)
    device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
    logger.info(f"Device chosen: {device}")

    # ---------------- DATA ----------------
    try:
        dataset = create_dataset(config)
    except Exception:
        logger.error(traceback.format_exc())
        raise

    logger.info(dataset)

    try:
        train_data, valid_data, test_data = data_preparation(config, dataset)
    except Exception:
        logger.error(traceback.format_exc())
        raise

    logger.info(f"[Training] batch={config['train_batch_size']} neg={config['train_neg_sample_args']}")
    logger.info(f"[Eval] batch={config['eval_batch_size']} args={config['eval_args']}")

    # ---------------- MODEL ----------------
    model = Lightgcn_price(config, dataset).to(device)
    logger.info(model)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainer = Trainer(config, model)
    best_model_path = os.path.join(log_dir, 'best_model.pth')

    # ---------------- TRAIN LOOP ----------------
    best_valid = -float('inf')

    for epoch in range(config['epochs']):
        t0 = time.time()
        train_loss = trainer._train_epoch(train_data, epoch, show_progress=config['show_progress'])
        logger.info(f"epoch {epoch} train [time={time.time()-t0:.2f}s loss={train_loss:.4f}]")

        if (epoch + 1) % config['eval_step'] == 0:
            e0 = time.time()
            valid_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=False)
            logger.info(f"epoch {epoch} eval [time={time.time()-e0:.2f}s]")
            logger.info(valid_result)

            valid_score = valid_result[config['valid_metric']]

            if valid_score > best_valid:
                best_valid = valid_score
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_score': float(best_valid)
                }, best_model_path)
                logger.info(f"New best saved (score={best_valid})")

    # ---------------- TEST ----------------
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])

        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
        logger.info("Test Result:")
        logger.info(test_result)
    else:
        logger.warning("No best model found → no test.")


if __name__ == '__main__':
    main()
