from collections import OrderedDict
import os.path as osp

import torch
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel


def _is_module_wrapper(module):
    return isinstance(module, (DataParallel, DistributedDataParallel))


def _warn(logger, msg):
    if logger is not None:
        logger.warning(msg)
    else:
        print(msg)


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(m, prefix=''):
        if _is_module_wrapper(m):
            m = m.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        m._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in m._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)

    missing_keys = [k for k in all_missing_keys if 'num_batches_tracked' not in k]

    if unexpected_keys:
        err_msg.append(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if strict and err_msg:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        msg = '\n'.join(err_msg)
        raise RuntimeError(msg)
    elif err_msg:
        _warn(logger, '\n'.join(err_msg))


def _load_checkpoint(filename, map_location=None):
    if filename.startswith(('http://', 'https://')):
        return torch.hub.load_state_dict_from_url(filename, map_location=map_location)
    if not osp.isfile(filename):
        raise IOError(f'{filename} is not a checkpoint file')
    return torch.load(filename, map_location=map_location)


def _strip_prefixes(state_dict):
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    if keys[0].startswith('module.'):
        state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

    keys = list(state_dict.keys())
    if keys and keys[0].startswith('backbone.'):
        print('Start stripping upper net pre-fix and loading backbone weights to our swin encoder')
        state_dict = OrderedDict((k.replace('backbone.', ''), v) for k, v in state_dict.items() if k.startswith('backbone.'))

    sorted_keys = sorted(state_dict.keys())
    if sorted_keys and sorted_keys[0].startswith('encoder.'):
        state_dict = OrderedDict((k.replace('encoder.', ''), v) for k, v in state_dict.items() if k.startswith('encoder.'))

    return state_dict


def _resize_position_embeddings(model, state_dict, logger=None):
    if state_dict.get('absolute_pos_embed') is not None and hasattr(model, 'absolute_pos_embed'):
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            _warn(logger, 'Error in loading absolute_pos_embed, pass')
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)

    model_state = model.state_dict()
    rel_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k and k in model_state]
    for key in rel_keys:
        table_pretrained = state_dict[key]
        table_current = model_state[key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            _warn(logger, f'Error in loading {key}, pass')
            continue
        if L1 != L2:
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            table_pretrained_resized = F.interpolate(
                table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                size=(S2, S2),
                mode='bicubic',
            )
            state_dict[key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    checkpoint = _load_checkpoint(filename, map_location)

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    state_dict = _strip_prefixes(state_dict)
    _resize_position_embeddings(model, state_dict, logger=logger)
    load_state_dict(model, state_dict, strict=strict, logger=logger)
    return checkpoint
