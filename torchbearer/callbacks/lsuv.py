# Copyright (C) 2017, Dmytro Mishkin
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file has been modified from https://github.com/ducha-aiki/LSUV-pytorch

from __future__ import print_function
import numpy as np
import torch
import torch.nn.init
import torch.nn as nn

gg = {}
gg['hook_position'] = 0
gg['total_fc_conv_layers'] = 0
gg['done_counter'] = -1
gg['hook'] = None
gg['act_dict'] = {}
gg['counter_to_apply_correction'] = 0
gg['correction_needed'] = False
gg['current_coef'] = 1.0


# Orthonorm init code is modified from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
def svd_orthonormal(w):
    shape = w.shape
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)#w;
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q.astype(np.float32)


def store_activations(self, input, output):
    gg['act_dict'] = output.data.cpu().numpy();
    return


def add_current_hook(m):
    if gg['hook'] is not None:
        return
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        if gg['hook_position'] > gg['done_counter']:
            gg['hook'] = m.register_forward_hook(store_activations)
        else:
            gg['hook_position'] += 1
    return


def count_conv_fc_layers(m):
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        gg['total_fc_conv_layers'] += 1
    return


def orthogonal_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        try:
            weight = m.weight_v
        except AttributeError:
            weight = m.weight

        w_ortho = svd_orthonormal(weight.data.cpu().numpy())
        m.weight.data = torch.from_numpy(w_ortho)
        try:
            nn.init.constant_(m.bias, 0)
        except Exception:
            pass


def apply_weights_correction(m):
    if gg['hook'] is None:
        return
    if not gg['correction_needed']:
        return
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        if gg['counter_to_apply_correction'] < gg['hook_position']:
            gg['counter_to_apply_correction'] += 1
        else:
            try:
                weight = m.weight_g
            except AttributeError:
                weight = m.weight
                
            weight.data *= float(gg['current_coef'])
            gg['correction_needed'] = False


def LSUVinit(model, data, needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = True):
    train = True if model.training else False

    model.eval()
    model.apply(count_conv_fc_layers)
    if do_orthonorm:
        model.apply(orthogonal_weights_init)
    for layer_idx in range(gg['total_fc_conv_layers']):
        model.apply(add_current_hook)
        out = model(data)
        current_std = gg['act_dict'].std()
        attempts = 0
        while (np.abs(current_std - needed_std) > std_tol):
            gg['current_coef'] =  needed_std / (current_std  + 1e-8);
            gg['correction_needed'] = True
            model.apply(apply_weights_correction)
            _ = model(data)
            current_std = gg['act_dict'].std()
            attempts+=1
            if attempts > max_attempts:
                break
        if gg['hook'] is not None:
           gg['hook'].remove()
        gg['done_counter']+=1
        gg['counter_to_apply_correction'] = 0
        gg['hook_position'] = 0
        gg['hook']  = None
          
    if train:
        model.train()
    return model