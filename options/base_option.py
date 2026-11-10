import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns", help='Name of this trial')

        self.parser.add_argument('--vq_name', type=str, default="rvq_nq1_dc512_nc512", help='Name of the rvq model.')

        self.parser.add_argument("--gpu_id", type=int, default=-1, help='GPU id')
        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name, {t2m} for humanml3d, {kit} for kit-ml')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here.')

        self.parser.add_argument('--latent_dim', type=int, default=1024, help='Dimension of transformer latent.') # 384 768 1536
        self.parser.add_argument('--n_heads', type=int, default=8, help='Number of heads.') # 6 8
        self.parser.add_argument('--n_layers', type=int, default=20, help='Number of attention layers.') # 8
        self.parser.add_argument('--ff_size', type=int, default=1024, help='FF_Size')
        self.parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio in transformer')

        self.parser.add_argument("--max_motion_length", type=int, default=196, help="Max length of motion")
        self.parser.add_argument("--unit_length", type=int, default=4, help="Downscale ratio of VQ")

        self.parser.add_argument('--force_mask', action="store_true", help='True: mask out conditions')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train

        if self.opt.gpu_id != -1:
            # self.opt.gpu_id = int(self.opt.gpu_id)
            torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if self.is_train:
            if self.opt.name != 'TEMP' and not getattr(self.opt, 'is_continue', False):
                import datetime
                # GATE0 fix F4: deterministic DDP run naming — all ranks must derive the SAME stamp
                # (launcher sets MASKCTRL_RUN_STAMP; per-rank datetime.now() can differ across ranks → split run dirs)
                run_stamp = os.environ.get('MASKCTRL_RUN_STAMP')
                if not run_stamp and int(os.environ.get('WORLD_SIZE', '1')) > 1:
                    # Fail loud: a per-rank datetime.now() fallback is nondeterministic across
                    # ranks and silently splits one run into multiple checkpoint dirs.
                    raise RuntimeError(
                        'Multi-rank run (WORLD_SIZE>1) requires MASKCTRL_RUN_STAMP to be set '
                        'by the launcher so all ranks derive the same run name, e.g. '
                        'export MASKCTRL_RUN_STAMP=$(date +%Y-%m-%d-%H-%M-%S) before torchrun.'
                    )
                date = run_stamp or datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                self.opt.name = f'z{date}_{self.opt.name}'
                # Re-capture after mutating opt.name so opt.txt records the stamped name
                # even if `args` above ever becomes a copy instead of a live view.
                args = vars(self.opt)
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir, exist_ok=True)
            # GATE0 fix F4: only rank 0 writes opt.txt (torchrun sets RANK; single-process runs default to 0)
            if int(os.environ.get('RANK', '0')) == 0:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        return self.opt