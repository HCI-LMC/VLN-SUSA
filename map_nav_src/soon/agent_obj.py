import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from reverie.agent_obj import GMapObjectNavAgent
from models.graph_utils import GraphMap
from models.model import VLNBert, Critic

def compute_outputs(embeds, attn):
    M = torch.tanh(embeds)
    a = torch.softmax(torch.matmul(M, attn), 1)  # [bs, max_len, 1]
    out = torch.sum(embeds * a, 1)  # [bs, hd]
    return torch.tanh(out)  # [bs, hd]

def compute_loss(outputs, txt_outputs, target_sim):
    sim = outputs @ txt_outputs.T
    return (F.cross_entropy(sim, target_sim, reduction='none') +
            F.cross_entropy(sim.T, target_sim.T, reduction='none')) / 2.0

class SoonGMapObjectNavAgent(GMapObjectNavAgent):

    def get_results(self):
        output = [{'instr_id': k, 
                    'trajectory': {
                        'path': v['path'], 
                        'obj_heading': [v['pred_obj_direction'][0]],
                        'obj_elevation': [v['pred_obj_direction'][1]],
                    }} for k, v in self.results.items()]
        return output

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)
        gdmaps = gmaps
        gsmaps = gmaps

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'pred_obj_direction': None,
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)
    
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.     
        og_loss = 0.   
        cl_loss = 0.

        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1
            for i, gdmap in enumerate(gdmaps):
                if not ended[i]:
                    gdmap.node_step_ids[obs[i]['viewpoint']] = t + 1
            for i, gsmap in enumerate(gsmaps):
                if not ended[i]:
                    gsmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)
            pano_depth_embeds, pano_depth_masks = self.vln_bert('depth_panorama',
                                                                pano_inputs)  # here 补充view的view_depth_fts
            avg_pano_depth_embeds = torch.sum(pano_depth_embeds * pano_depth_masks.unsqueeze(2), 1) / \
                                    torch.sum(pano_depth_masks, 1, keepdim=True)  # here--------------------------

            pano_sem_embeds, pano_sem_masks = self.vln_bert('sem_panorama', pano_inputs)  # here 补充view的view_sem_fts
            avg_pano_sem_embeds = torch.sum(pano_sem_embeds * pano_sem_masks.unsqueeze(2), 1) / \
                                  torch.sum(pano_sem_masks, 1, keepdim=True)  # here-------------------------

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            for i, gdmap in enumerate(gdmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gdmap.update_node_embed(i_vp, avg_pano_depth_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gdmap.graph.visited(i_cand_vp):
                            gdmap.update_node_embed(i_cand_vp, pano_depth_embeds[i, j])
            for i, gsmap in enumerate(gsmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gsmap.update_node_embed(i_vp, avg_pano_sem_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gsmap.graph.visited(i_cand_vp):
                            gsmap.update_node_embed(i_cand_vp, pano_sem_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(self._nav_gdmap_variable(obs, gdmaps))  # here----
            nav_inputs.update(self._nav_gsmap_variable(obs, gsmaps))  # here----

            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_depth_embeds, pano_sem_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['obj_lens'], 
                    pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
            })
            nav_outs = self.vln_bert('navigation', nav_inputs)


            nav_logits = nav_outs['fused_logits']
            nav_vpids = nav_inputs['gmap_vpids']
            gmap_embeds = nav_outs['gmap_embeds']
            gmap_depth_embeds = nav_outs['gmap_depth_embeds']
            vp_embeds = nav_outs['vp_embeds']
            vp_sem_embeds = nav_outs['vp_sem_embeds']
            fw_global = nav_outs['fw_global']
            fw_local = nav_outs['fw_local']
            fw_global_depth = nav_outs['fw_global_depth']
            fw_local_sem = nav_outs['fw_local_sem']
            txt_encoder_embeds = nav_inputs['txt_embeds']

            nav_probs = torch.softmax(nav_logits, 1)
            obj_logits = nav_outs['obj_logits']
            
            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    # update i_vp: stop and object grounding scores
                    i_objids = obs[i]['obj_ids']
                    i_obj_logits = obj_logits[i, pano_inputs['view_lens'][i]+1:]
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                        'og': i_objids[torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                        'og_direction': obs[i]['obj_directions'][torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                        'og_details': {'objids': i_objids, 'logits': i_obj_logits[:len(i_objids)]},
                    }
                                        
            if train_ml is not None:
                # Supervised training
                nav_targets = self._teacher_action(
                    obs, nav_vpids, ended, 
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
                )
                # print(t, nav_logits, nav_targets)
                ml_loss += self.criterion(nav_logits, nav_targets)
                # print(t, 'ml_loss', ml_loss.item(), self.criterion(nav_logits, nav_targets).item())
                if self.args.fusion in ['avg', 'dynamic'] and self.args.loss_nav_3:
                    # add global and local losses
                    ml_loss += self.criterion(nav_outs['global_logits'], nav_targets)    # global
                    local_nav_targets = self._teacher_action(
                        obs, nav_inputs['vp_cand_vpids'], ended, visited_masks=None
                    )
                    ml_loss += self.criterion(nav_outs['local_logits'], local_nav_targets)  # local
                # objec grounding 
                obj_targets = self._teacher_object(obs, ended, pano_inputs['view_lens'])
                # print(t, obj_targets[6], obj_logits[6], obs[6]['obj_ids'], pano_inputs['view_lens'][i], obs[6]['gt_obj_id'])
                og_loss += self.criterion(obj_logits, obj_targets)
                # print(F.cross_entropy(obj_logits, obj_targets, reduction='none'))
                # print(t, 'og_loss', og_loss.item(), self.criterion(obj_logits, obj_targets).item())

                ########################### cl loss ##################################

                txt_outputs = compute_outputs(txt_encoder_embeds, self.tim_txt_attn)
                target_sim = torch.arange(batch_size).cuda()

                gmap_outputs = compute_outputs(gmap_embeds, self.tim_global_img_attn)
                gmap_depth_outputs = compute_outputs(gmap_depth_embeds, self.tim_global_depth_attn)
                vp_outputs = compute_outputs(vp_embeds, self.tim_local_img_attn)
                vp_sem_outputs = compute_outputs(vp_sem_embeds, self.tim_local_sem_attn)
                fused_outputs = fw_global * gmap_outputs + fw_local * vp_outputs + fw_global_depth * gmap_depth_outputs + fw_local_sem * vp_sem_outputs
                cl_loss += compute_loss(fused_outputs, txt_outputs, target_sim).mean() * 0.8


                                                   
            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach() 
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])   

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf'), 'og': None}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    traj[i]['pred_obj_direction'] = stop_score['og_direction']
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                                'obj_ids': [str(x) for x in v['og_details']['objids']],
                                'obj_logits': v['og_details']['logits'].tolist(),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            og_loss = og_loss * train_ml / batch_size
            cl_loss = cl_loss * train_ml / batch_size
            self.loss += ml_loss
            self.loss += og_loss
            self.loss += cl_loss
            self.logs['IL_loss'].append(ml_loss.item())
            self.logs['OG_loss'].append(og_loss.item())
            self.logs['CL_loss'].append(cl_loss.item())

        return traj
              
    