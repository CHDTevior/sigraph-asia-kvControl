from data.t2m_dataset import Text2MotionDatasetEval, collate_fn # TODO
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt

import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path


def resolve_stat_path(explicit_path, opt, field_name, default_name):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    opt_value = getattr(opt, field_name, None)
    if opt_value:
        candidates.append(Path(opt_value))
    meta_dir = getattr(opt, "meta_dir", None)
    if meta_dir:
        candidates.append(Path(meta_dir) / default_name)
    data_root = getattr(opt, "data_root", None)
    if data_root:
        candidates.append(Path(data_root) / default_name.replace(".npy", ".npy").replace("mean.npy", "Mean.npy").replace("std.npy", "Std.npy"))

    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        f"Unable to resolve {field_name}. Checked: {[str(c) for c in candidates]}"
    )


def get_dataset_motion_loader(opt_path, batch_size, fname, device, mean_path=None, std_path=None):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(resolve_stat_path(mean_path, opt, 'mean_npy', 'mean.npy'))
        std = np.load(resolve_stat_path(std_path, opt, 'std_npy', 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        # BS1_LATENCY opt-in (default OFF -> identical to prior behavior): the
        # batch=1 latency harness needs exactly one sample per generation call
        # and a fixed, reproducible sample order (no shuffling). Every existing
        # (non-opt-in) caller keeps batch_size as passed and shuffle=True.
        if os.environ.get('BS1_LATENCY') == '1':
            batch_size = 1
            shuffle = False
        else:
            shuffle = True
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=shuffle)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset
