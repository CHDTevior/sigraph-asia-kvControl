#!/usr/bin/env python3
"""Audit: count public MaskControl ControlNet mechanism trainable params (paper \paramsCtrlnetPub).
Loads the released MaskControl trajectory CtrlNet ckpt and sums the control-added modules
(duplicated transformer branch + control encoder + zero-conv connectors), excluding the
frozen base transformer and the frozen VQ tokenizer. Result: 21.372M ~= 21.4M (2026-07-08).
Usage: python scripts/audit_maskcontrol_ctrlnet_params.py <ckpt.tar>
"""
import torch, sys, collections
CK = sys.argv[1] if len(sys.argv) > 1 else \
  "/iridisfs/scratch/ts1v23/workspace/competitors/ckpts/maskcontrol_trajectory/z2024-08-23-01-27-51_CtrlNet_randCond1-196_l1.1XEnt.9TTT__fixRandCond/model/net_best_acc.tar"
sd = torch.load(CK, map_location='cpu', weights_only=False)['ct2m_transformer']
CTRL = ['seqTransEncoder_control', 'encoder_control', 'first_zero_linear', 'mid_zero_linear']
per = collections.OrderedDict((p, sum(v.numel() for k,v in sd.items()
        if k.startswith(p+'.') and hasattr(v,'shape'))) for p in CTRL)
for p,n in per.items(): print(f"  {p:26} {n/1e6:8.3f}M")
print(f"  {'= ControlNet mechanism':26} {sum(per.values())/1e6:8.3f}M  (paper \\paramsCtrlnetPub = 21.4M)")
