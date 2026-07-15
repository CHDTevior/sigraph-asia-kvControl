import os

import clip
import numpy as np
import torch
# from scipy import linalg
from utils.metrics import *
import torch.nn.functional as F
# import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric
from models.mask_transformer.tools import *
from common.quaternion import qrot, qinv
from einops import rearrange, repeat
import random
from utils.metrics import random_mask_cross
from models.mask_transformer.control_transformer import ControlTransformer
#
#
# def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
#     xyz = xyz[:1]
#     bs, seq = xyz.shape[:2]
#     xyz = xyz.reshape(bs, seq, -1, 3)
#     plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
#     plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
#     writer.add_video(tag, plot_xyz, nb_iter, fps=20)
from data_loaders.humanml.utils.metrics import calculate_skating_ratio, compute_kps_error, calculate_trajectory_error, calculate_trajectory_diversity


@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, mpjpe

@torch.no_grad()
def evaluation_vqvae_plus_l1(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"%\
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_res_plus_l1(val_loader, vq_model, res_model, repeat_id, eval_wrapper, num_joint, do_vq_res=True):
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        if do_vq_res:
            code_ids, all_codes = vq_model.encode(motion)
            if len(code_ids.shape) == 3:
                pred_vq_codes = res_model(code_ids[..., 0])
            else:
                pred_vq_codes = res_model(code_ids)
            # pred_vq_codes = pred_vq_codes - pred_vq_res + all_codes[1:].sum(0)
            pred_pose_eval = vq_model.decoder(pred_vq_codes)
        else:
            rec_motions, _, _ = vq_model(motion)
            pred_pose_eval = res_model(rec_motions)        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"%\
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist

@torch.no_grad()
def evaluation_mask_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False,num_repeat=1,rand_pos=False, CFG=-1):
    

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    
    time_steps = 18
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 3.25

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)
    _mean = torch.from_numpy(val_loader.dataset.mean).cuda()
    _std = torch.from_numpy(val_loader.dataset.std).cuda()

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()
        # print(f"eval 时候的motion shape:{pose.shape}")

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22
        pred_len = m_length.cuda()
        m_tokens_len = torch.ceil((m_length)/4)
        pred_tok_len = m_tokens_len

        # (b, seqlen)
        # print(f"eval 时候的m_length:{m_length}")
        mids, logits  = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1) # mids 是 list. 6 * [b,49]
        
        
        mids = [mids[i].unsqueeze(-1) for i in range(len(mids))]  # 6 * [b,49,1]
        mids_torch = torch.stack(mids, dim=-1)  # torch.Size([B, 49, 1, 6])
        mids_torch = mids_torch.squeeze(2)  # torch.Size([B, 49, 6])
        # motion_codes = motion_codes.permute(0, 2, 1)
        pred_motions = torch.zeros((bs, seq, pose.shape[-1])).cuda()

        # 检查下vqvae的
        
        pose = pose.cuda().float()
        for k in range(bs): 
            # pred_motions_vae,_,_ = vq_model(pose[k:k+1,:int(m_length[k])]) 
            pred_pose = vq_model.vqvae.forward_decoder(mids_torch[k:k+1,:int(pred_tok_len[k].item())])  # torch.Size([1, 196, 263])
            pred_motions[k:k+1,:int(pred_len[k])] = pred_pose
        # pred_motions = vq_model.vqvae.forward_decoder(mids_torch)  # torch.Size([B, 196, 263])

        # import pdb; pdb.set_trace()   

        ################ 注释掉直接用mids拿码本 ####################
        # seq_len = 49
        # m_lens = m_length // 4
        # padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # # emb = F.softmax(logits/1, dim=-1) @ vq_model.quantizer.codebooks[0]

        # emb_list= [F.softmax(logits[:,i,:,:]/1, dim=-1) @ vq_model.vqvae.quantizers[i].codebook for i in range(6)] # 6 * [512,49,128]
        # # import pdb; pdb.set_trace()

        # # emb = emb.masked_fill(padding_mask.unsqueeze(-1), 0.)
        # emb_list = [emb_list[i].masked_fill(padding_mask.unsqueeze(-1), 0.) for i in range(6)]  # 6 * [512,49,128]
        # emb_list = [emb_list[i].permute(0,2,1) for i in range(6)] # 6 * [512,128,49]


        # # pred_motions = vq_model.forward_decoder(emb)
        # pred_motions = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list) #torch.Size([B, 263, 1, 196])
        # pred_motions = pred_motions.squeeze(2).permute(0, 2, 1) # torch.Size([B, 196, 263])
        #################################################

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    # GATE0 fix F5: TB writes must never kill training (a shared-fs FileNotFoundError killed May segment B)
    if writer is not None:
        try:
            writer.add_scalar('./Test/FID', fid, ep)
            writer.add_scalar('./Test/Diversity', diversity, ep)
            writer.add_scalar('./Test/top1', R_precision[0], ep)
            writer.add_scalar('./Test/top2', R_precision[1], ep)
            writer.add_scalar('./Test/top3', R_precision[2], ep)
            writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        except (FileNotFoundError, OSError) as _tb_err:
            print(f"  WARNING: TensorBoard write failed (non-fatal): {_tb_err}")


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_top3.tar'), ep)

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_res_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(code_indices[..., 0], clip_text, m_length//4,
                                      temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    # GATE0 fix F5: TB writes must never kill training (a shared-fs FileNotFoundError killed May segment B)
    if writer is not None:
        try:
            writer.add_scalar('./Test/FID', fid, ep)
            writer.add_scalar('./Test/Diversity', diversity, ep)
            writer.add_scalar('./Test/top1', R_precision[0], ep)
            writer.add_scalar('./Test/top2', R_precision[1], ep)
            writer.add_scalar('./Test/top3', R_precision[2], ep)
            writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        except (FileNotFoundError, OSError) as _tb_err:
            print(f"  WARNING: TensorBoard write failed (non-fatal): {_tb_err}")


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_top3.tar'), ep)

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_res_transformer_plus_l1(val_loader, vq_model, trans, repeat_id, eval_wrapper, num_joint,
                                       cond_scale=2, temperature=1, topkr=0.9, cal_l1=True):


    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    l1_dist = 0
    num_poses = 1
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # print(code_indices[0:2, :, 1])

        pred_ids = trans.generate(code_indices[..., 0], clip_text, m_length//4, topk_filter_thres=topkr,
                                  temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        if cal_l1:
            bgt = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
            bpred = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
            for i in range(bs):
                gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
                pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
                # gt = motion[i, :m_length[i]]
                # pred = pred_pose_eval[i, :m_length[i]]
                num_pose = gt.shape[0]
                l1_dist += F.l1_loss(gt, pred) * num_pose
                num_poses += num_pose

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_mask_transformer_test(val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False, cal_mm=True):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        # print(i)
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                mids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            mids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(mids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality

traj_err_key = ["traj_fail_20cm", "traj_fail_50cm (Traj err)", "kps_fail_20cm", "kps_fail_50cm (Loc. err)", "kps_mean_err(m) (Avg. err)"]

# @torch.no_grad()
def evaluation_mask_transformer_test_plus_res(val_loader, vq_model, res_model, ct2m_transformer, rt2m_transformer, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5, f=None, pred_num_batch=30, logger=None, epoch=-1, control='trajectory', density=5, opt=None):
    cal_mm = False

    ct2m_transformer.eval()
    if rt2m_transformer is not None:
        rt2m_transformer.eval()
    vq_model.eval()
    if res_model is not None:
        res_model.eval()
    torch.set_grad_enabled(True) # ===> or remove "@torch.no_grad()"

    # Infer device from model to support multi-GPU (--gpu_id > 0)
    _model_device = next(ct2m_transformer.parameters()).device

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    skate_ratio = 0
    multimodality = 0
    traj_err = []
    avoid_dist = 0

    _mean = torch.from_numpy(val_loader.dataset.mean).to(_model_device)
    _std = torch.from_numpy(val_loader.dataset.std).to(_model_device)

    for param in ct2m_transformer.parameters():
        param.requires_grad = False
    if rt2m_transformer is not None:
        for param in rt2m_transformer.parameters():
            param.requires_grad = False
    for param in vq_model.parameters():
        param.requires_grad = False

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    
    #### R-Precision only works with batch 32
    def batches32(*seqs):
        b = 32
        l = len(seqs[0])
        for ndx in range(0, l, b):
            yield [seq[ndx:min(ndx + b, l)] for seq in seqs]
    batch32_nb_sample = 0

    from tqdm import tqdm
    all_speed = []
    for i, batch in enumerate(tqdm(val_loader)):
        if i % pred_num_batch == 0:
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = [], [], (), [], [], [], '__token__'
        _word_embeddings, _pos_one_hots, _clip_text, _sent_len, _pose, _m_length, _token = batch
        word_embeddings.append(_word_embeddings)
        pos_one_hots.append(_pos_one_hots)
        clip_text = clip_text + _clip_text
        sent_len.append(_sent_len)
        pose.append(_pose)
        m_length.append(_m_length)
        
        if (i+1) % pred_num_batch != 0:
            continue
        else:
            word_embeddings = torch.cat(word_embeddings)
            pos_one_hots = torch.cat(pos_one_hots)
            sent_len = torch.cat(sent_len)
            pose = torch.cat(pose)
            m_length = torch.cat(m_length)
        

        m_length = m_length.to(_model_device)
        pose = pose.to(_model_device).float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        global_joint = recover_from_ric((pose * _std + _mean).float(), opt.joints_num)

        ##### Random Per Sample #####
        global_joint_mask = []
        for i in range(m_length.shape[0]):
            # fixed positive density reuses the same count-scaling rule as the mixture
            dens = random.choice([1,2,5,49,196]) if density == -1 else density

            if dens in [1, 2, 5]:
                rand_dens = dens
            else:
                rand_dens = int(m_length[i] * (dens/196))
            selected_frames = np.random.choice(m_length[i].detach().cpu().numpy(), rand_dens, replace=False)
            # selected_frames = np.arange(0, rand_dens)
            mask_frame = np.zeros((1, pose.shape[1]))
            mask_frame[0, selected_frames] = 1
            global_joint_mask.append(mask_frame)
        global_joint_mask = torch.tensor(np.concatenate(global_joint_mask), device=pose.device, dtype=bool)

        ##### Random Per batch #####
        # all_len_mask = lengths_to_mask(m_length, 196) #(b, n) # using "pose.shape[1]" can be lower than 196
        # batch_randperm = torch.rand((bs, pose.shape[1]), device=pose.device)
        # batch_randperm[~all_len_mask] = 1
        # batch_randperm = batch_randperm.argsort(dim=-1)
        # global_joint_mask = batch_randperm < (density if density != -1 else random.choice([1,2,5,49,196]))
        # global_joint_mask = global_joint_mask * all_len_mask
        ######################################
        if control == 'trajectory' or control == 'pelvis':
            global_joint_mask = repeat(global_joint_mask, 'b f -> b f j', j=opt.joints_num).clone()
            global_joint_mask[..., 1:] = False
        elif control == 'cross':
            cross_joints = cross_combination_joints()
            choose = np.random.choice(len(cross_joints), 1).item()
            choose_joint = cross_joints[choose]
            _global_joint_mask = global_joint_mask
            global_joint_mask = torch.zeros((*global_joint_mask.shape, opt.joints_num), device=_global_joint_mask.device, dtype=bool)
            global_joint_mask[..., choose_joint] = _global_joint_mask.unsqueeze(-1)
        elif control in ['pelvis', 'l_foot', 'r_foot', 'head', 'left_wrist', 'right_wrist', 'lower']:
            controllable_joints = {
                "pelvis": 0,
                "l_foot": 10,
                "r_foot": 11,
                "head": 15,
                "left_wrist": 20,
                "right_wrist": 21,
                "lower": [0, 10, 11]
            }
            _global_joint_mask = global_joint_mask
            if control == 'lower':
                _global_joint_mask = _global_joint_mask.unsqueeze(-1)
            global_joint_mask = torch.zeros((*global_joint_mask.shape, opt.joints_num), device=_global_joint_mask.device, dtype=bool)
            global_joint_mask[..., controllable_joints[control]] = _global_joint_mask
        elif control == 'random':
            _global_joint_mask = global_joint_mask
            global_joint_mask = torch.zeros((*global_joint_mask.shape, opt.joints_num), device=_global_joint_mask.device, dtype=bool)
            control_joints = torch.tensor([0, 10, 11, 15, 20, 21], device=pose.device)
            rand_indx = torch.randint(len(control_joints), (_global_joint_mask.shape[0],)) # random index (bs,)
            global_joint_mask[torch.arange(global_joint_mask.shape[0]),:, 
                            control_joints[rand_indx]] = _global_joint_mask # set idx of joint to frames mask
        elif control == 'all':
            _global_joint_mask = global_joint_mask
            global_joint_mask = torch.zeros((*global_joint_mask.shape, opt.joints_num), device=_global_joint_mask.device, dtype=bool)
            control_joints = torch.tensor([0, 10, 11, 15, 20, 21], device=pose.device)
            global_joint_mask[..., control_joints] = _global_joint_mask.unsqueeze(-1)
        else:
            raise Exception(f'{control} is not implemented yet!!!')
        ######################################
        avoid_points=None
        if False:
            global_joint_mask = torch.zeros(global_joint.shape[:-1]).to(global_joint.device).bool()
            # global_joint_mask[torch.arange(m_length.shape[0]), (m_length-1), 0] = True
            avoid_points = torch.zeros([m_length.shape[0], 196, 4]).cuda()
            end_f = torch.floor(m_length/1.).long()-1
            # add size of sdf to the index 3 of frames > 20
            # m = lengths_to_mask(m_length-20, 196) # remove last 20 frames
            # m[:, :20] = False # remove first 20 frames
            # m = repeat(m, 'b f -> b f d', d=4).clone()
            # m[..., :3] = False
            # # all frames [20:-20] avoid only the point of the middle from gt
            # avoid_points[:, :, :3] = global_joint[torch.arange(m_length.shape[0]), mid_f, 0:1]
            # avoid_points[m] = .25
            batch_select = torch.arange(m_length.shape[0])
            avoid_points[batch_select, end_f, :3] = global_joint[batch_select, 0, 0]
            avoid_points[batch_select, end_f, 3] = 2
            
        import timeit
        start = timeit.default_timer()
        from models.mask_transformer.control_transformer_my import ControlTransformer
        print(f"ct2m_transformer 的type是: {type(ct2m_transformer)},{type(ct2m_transformer) is ControlTransformer}")
        if type(ct2m_transformer) is ControlTransformer:
            ct2m_transformer.ctrl_net = opt.ctrl_net
            ct2m_transformer.rt2m_transformer = rt2m_transformer
            pred_motions_denorm, pred_motions = ct2m_transformer.generate_with_control(clip_text, m_length, time_steps, cond_scale,
                                                                            temperature=temperature, topkr=topkr,
                                                                            force_mask=force_mask,
                                                                            vq_model=vq_model,
                                                                            global_joint=global_joint,
                                                                            global_joint_mask=global_joint_mask,
                                                                            _mean=_mean,
                                                                            _std=_std,
                                                                            res_cond_scale=res_cond_scale,
                                                                            res_model=res_model,
                                                                            control_opt = {
                                                                                'each_lr': opt.each_lr,
                                                                                'each_iter': opt.each_iter,
                                                                                'lr': opt.last_lr,
                                                                                'iter': opt.last_iter
                                                                            },
                                                                            avoid_points=avoid_points)
        # GN replacement path: the Gauss-Newton Stage-2 is batch=1 (per-sample Jacobians), so
        # generation runs sample-by-sample and the group is reassembled IN ORDER before the
        # unchanged metric pipeline (codex 019f59cc: keep the 32-sample retrieval groups and
        # negative pools identical; only generation is sequential).
        elif ((getattr(opt, 'gn_cfg', None) is not None and not getattr(opt, 'gn_batch', False))
                or getattr(opt, 'seq_gen', False)) \
                and hasattr(ct2m_transformer, 'generate_with_control'):
            ct2m_transformer.ctrl_net = opt.ctrl_net
            _B = len(clip_text)
            _T_out = int(4 * (m_length.max().item() // 4))
            _pmd_parts, _pm_parts = [], []
            for _b in range(_B):
                # trim the control tensors to THIS sample's effective length — the batched
                # buffers are sized by the batch max (196), the per-sample generation by
                # 4*(m_length[b]//4), and _build_ctrl_cond subtracts them elementwise
                _T_b = int(4 * (m_length[_b].item() // 4))
                _pmd_b, _pm_b = ct2m_transformer.generate_with_control(
                    [clip_text[_b]], m_length[_b:_b+1], time_steps, cond_scale,
                    temperature=temperature, topkr=topkr, force_mask=force_mask,
                    vq_model=vq_model,
                    global_joint=global_joint[_b:_b+1, :_T_b],
                    global_joint_mask=global_joint_mask[_b:_b+1, :_T_b],
                    _mean=_mean, _std=_std,
                    control_opt={
                        'each_lr': opt.each_lr, 'each_iter': opt.each_iter,
                        'lr': getattr(opt, 'last_lr', 6e-2),
                        'iter': getattr(opt, 'last_iter', 0),
                        'rgar': getattr(opt, 'rgar_cfg', None),
                        's2_optimizer': ('gn' if getattr(opt, 'gn_cfg', None) is not None
                                         else None),
                        'gn': getattr(opt, 'gn_cfg', None),
                    },
                    avoid_points=avoid_points)
                # pad the per-sample T up to the group's T; downstream metrics truncate by
                # m_length, so the pad content is never read
                if _pmd_b.shape[1] < _T_out:
                    _pad = _T_out - _pmd_b.shape[1]
                    _pmd_b = torch.cat([_pmd_b, _pmd_b[:, -1:].expand(-1, _pad, -1, -1)], dim=1)
                    _pm_b = torch.cat([_pm_b, _pm_b[:, -1:].expand(-1, _pad, -1)], dim=1)
                _pmd_parts.append(_pmd_b)
                _pm_parts.append(_pm_b)
            pred_motions_denorm = torch.cat(_pmd_parts, dim=0)
            pred_motions = torch.cat(_pm_parts, dim=0)
        # GATE0 fix F1: dispatch on capability (generate_with_control), not exact class identity
        elif hasattr(ct2m_transformer, 'generate_with_control'):
            ct2m_transformer.ctrl_net = opt.ctrl_net
            pred_motions_denorm, pred_motions = ct2m_transformer.generate_with_control(clip_text, m_length, time_steps, cond_scale,
                                                                            temperature=temperature, topkr=topkr,
                                                                            force_mask=force_mask,
                                                                            vq_model=vq_model,
                                                                            global_joint=global_joint,
                                                                            global_joint_mask=global_joint_mask,
                                                                            _mean=_mean,
                                                                            _std=_std,
                                                                            control_opt={
                                                                                'each_lr': opt.each_lr,
                                                                                'each_iter': opt.each_iter,
                                                                                'lr': getattr(opt, 'last_lr', 6e-2),
                                                                                'iter': getattr(opt, 'last_iter', 0),
                                                                                'rgar': getattr(opt, 'rgar_cfg', None),
                                                                                # gn_batch: batched Stage-1 + per-sample GN Stage-2
                                                                                's2_optimizer': ('gn' if (getattr(opt, 'gn_cfg', None) is not None
                                                                                                 and getattr(opt, 'gn_batch', False)) else None),
                                                                                'gn': (getattr(opt, 'gn_cfg', None)
                                                                                       if getattr(opt, 'gn_batch', False) else None),
                                                                            },
                                                                            avoid_points=avoid_points)
        else:
            # GATE0 fix F1: control eval must never silently fall through to uncontrolled generate_momask
            raise RuntimeError(
                f"Control eval requires generate_with_control, but {type(ct2m_transformer).__name__} "
                f"does not provide it — refusing to silently fall back to generate_momask")

        stop = timeit.default_timer()
        speed = stop - start
        all_speed.append(speed)
        print('Time: ', speed, 'avg:', np.array(all_speed).mean())  
        for _word_embeddings, _pos_one_hots, _sent_len, _pose, _m_length, _pred_motions in \
            batches32(word_embeddings, pos_one_hots, sent_len, pose, m_length, pred_motions):
            if len(_word_embeddings) != 32:
                break

            et_pred, em_pred = eval_wrapper.get_co_embeddings(_word_embeddings, _pos_one_hots, _sent_len,
                                                                _pred_motions.clone(),
                                                                _m_length)

            et, em = eval_wrapper.get_co_embeddings(_word_embeddings, _pos_one_hots, _sent_len, _pose, _m_length)
            motion_annotation_list.append(em)
            motion_pred_list.append(em_pred)

            temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
            temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
            R_precision_real += temp_R
            matching_score_real += temp_match
            # print(et_pred.shape, em_pred.shape)
            temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
            temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
            R_precision += temp_R
            matching_score_pred += temp_match

            batch32_nb_sample += len(_word_embeddings)

        if opt.dataset_name == 'kit':
            pred_motions_denorm = pred_motions_denorm / 1000 * 2
            global_joint = global_joint / 1000 * 2
        if avoid_points is not None:
            # query = avoid_points[..., 3] > 0
            # dist = torch.norm(avoid_points[..., :3][query] - pred_motions_denorm[:, :, 0][query], dim=-1)
            # dist = torch.clamp(avoid_points[..., 3][query] - dist, min=0.0)
            # avoid_dist += dist.sum().detach().cpu().numpy()
            
            dist = torch.norm(avoid_points[..., :3] - pred_motions_denorm[:, :, 0], dim=-1)
            dist = torch.clamp(avoid_points[..., 3] - dist, min=0.0)
            dist = dist.sum(-1) / ((avoid_points[..., 3] > 0).sum(-1) + 1e-8)
            avoid_dist += dist.sum().detach().cpu().numpy()
        
        temp_skate_ratio, skate_vel = calculate_skating_ratio(pred_motions_denorm.permute(0, 2, 3, 1)) # => [32, 22, 3, 196]
        skate_ratio += temp_skate_ratio.sum()

        kps_error, t_mask = compute_kps_error(pred_motions_denorm.permute(0, 2, 3, 1), 
                          global_joint.permute(0, 2, 3, 1), 
                          repeat(global_joint_mask, 'b f j -> b j d f', d=3))
        err_np = calculate_trajectory_error(kps_error, t_mask)
        traj_err.append(err_np)

        nb_sample += bs
        # if nb_sample > 400:
        #     break

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if batch32_nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if batch32_nb_sample > 300 else 100)

    R_precision_real = R_precision_real / batch32_nb_sample
    R_precision = R_precision / batch32_nb_sample

    matching_score_real = matching_score_real / batch32_nb_sample
    matching_score_pred = matching_score_pred / batch32_nb_sample
    skate_ratio = skate_ratio / nb_sample
    avoid_dist = avoid_dist / nb_sample
    traj_err = np.stack(traj_err).reshape((-1, traj_err[0].shape[-1])).mean(0)
    line = ''
    for (k, v) in zip(traj_err_key, traj_err):
        line += '(%s): %.4f ' % (k, np.mean(v))

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}, " \
          f"skate_ratio. {skate_ratio:.4f}, " \
          f"avoid_dist. {avoid_dist:.4f}, " \
          f"multimodality. {multimodality:.4f}, \n" \
          f"{line}"
    print(msg)
    if f is not None:
        print(msg, file=f, flush=True)
    if logger is not None:
        # GATE0 fix F5: eval-logger writes must never kill training (shared-fs hiccups raise FileNotFoundError/OSError)
        try:
            logger.add_scalar('./Val/FID', fid, epoch)
            logger.add_scalar('./Val/Diversity', diversity, epoch)
            logger.add_scalar('./Val/top1', R_precision[0], epoch)
            logger.add_scalar('./Val/top2', R_precision[1], epoch)
            logger.add_scalar('./Val/top3', R_precision[2], epoch)
            logger.add_scalar('./Val/matching_score', matching_score_pred, epoch)
            logger.add_scalar('./Val/skate_ratio', skate_ratio, epoch)
            logger.add_scalar('./Val/avoid_dist', avoid_dist, epoch)
            for (k, v) in zip(traj_err_key, traj_err):
                logger.add_scalar('./Val/'+k, v, epoch)
        except (FileNotFoundError, OSError) as _tb_err:
            print(f"  WARNING: eval logger write failed (non-fatal): {_tb_err}")
    
    kps_mean = traj_err[traj_err_key.index('kps_mean_err(m) (Avg. err)')]
    return fid, diversity, R_precision, matching_score_pred, skate_ratio, multimodality, traj_err, avoid_dist, kps_mean

def _recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    # r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos_glob = torch.cumsum(r_pos.detach(), dim=1)
    _r_pos_glob = torch.cat((torch.zeros_like(r_pos_glob[:, :1]), r_pos_glob[:, :-1]), dim=1)
    r_pos = _r_pos_glob + r_pos

    r_pos[..., 1] = data[..., 3]
    return r_rot_ang, r_pos