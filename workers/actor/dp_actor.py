# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=True, return_logits=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        重构后的前向传播方法，始终返回 logits 以保证计算图的稳定。

        Returns:
            A tuple of:
            - entropy: (bs, response_len) or None
            - log_probs: (bs, response_len)
            - logits: (bs, response_len, vocab_size)
        """
        calculate_entropy = (self.config.entropy_coeff != 0) and calculate_entropy
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(input_ids_rmpad, position_ids_rmpad=position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad=position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                
                output = self.actor_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, **multi_modal_inputs, use_cache=False)
                logits_rmpad = output.logits.squeeze(0)

                if self.use_ulysses_sp:
                    logits_rmpad = gather_outpus_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                full_logits = pad_input(hidden_states=logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                
                logits = full_logits[:, -response_length - 1 : -1, :]

            else: # not using rmpad
                output = self.actor_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **multi_modal_inputs, use_cache=False)
                logits_full = output.logits
                logits = logits_full[:, -response_length - 1 : -1, :]

            logits.div_(temperature)
            
            # 始终使用 False 来保证梯度流
            log_probs = logprobs_from_logits(logits, micro_batch["responses"], inplace_backward=False)

            # 仅在需要时计算 entropy
            # entropy = None
            # if calculate_entropy:
            #     if not self.config.entropy_checkpointing:
            #         entropy = self.compute_entropy_from_logits(logits)
            #     else:
            #         entropy = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits)
                    
            entropy = self.compute_entropy_from_logits(logits)

            return entropy, log_probs, logits

    # def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, return_logits=False) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Returns:
    #         entropy: # (bs, response_len)
    #         log_probs: # (bs, response_len)
    #     """
    #     response_length = micro_batch["responses"].size(-1)
    #     multi_modal_inputs = {}
    #     if "multi_modal_inputs" in micro_batch:
    #         for key in micro_batch["multi_modal_inputs"][0].keys():
    #             multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

    #     entropy = None
    #     logits_for_logprob = None

    #     with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
    #         input_ids = micro_batch["input_ids"]
    #         batch_size, seqlen = input_ids.shape
    #         attention_mask = micro_batch["attention_mask"]
    #         position_ids = micro_batch["position_ids"]
    #         entropy = None
    #         if position_ids.dim() == 3:  # qwen2vl mrope
    #             position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

    #         if self.use_remove_padding:
    #             input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
    #             input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

    #             # unpad the position_ids to align the rotary
    #             if position_ids.dim() == 3:
    #                 position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
    #             else:
    #                 position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

    #             # for compute the log_prob
    #             input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

    #             if return_logits and not self.use_fused_kernels:
    #                 if self.use_ulysses_sp:
    #                     logits_rmpad_gathered = gather_outpus_and_unpad(
    #                         logits_rmpad,
    #                         gather_dim=0,
    #                         unpad_dim=0,
    #                         padding_size=pad_size,
    #                     )
    #                 else:
    #                     logits_rmpad_gathered = logits_rmpad

    #                 # 将 logits 还原回原始的 (bsz, seqlen, vocab_size) 形状
    #                 full_logits = pad_input(
    #                     hidden_states=logits_rmpad_gathered,
    #                     indices=indices,
    #                     batch=batch_size,
    #                     seqlen=seqlen,
    #                 )
    #                 # 切片以获取响应部分的 logits
    #                 logits_for_logprob = full_logits[:, -response_length - 1 : -1, :]

    #             # pad and slice the inputs if sp > 1
    #             if self.use_ulysses_sp:
    #                 is_vlm_model = "multi_modal_inputs" in micro_batch
    #                 if is_vlm_model:
    #                     # vlm model's inputs will be sliced after embedding
    #                     input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
    #                         input_ids_rmpad,
    #                         position_ids_rmpad=position_ids_rmpad,
    #                         sp_size=self.ulysses_sequence_parallel_size,
    #                     )
    #                 else:
    #                     input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
    #                         input_ids_rmpad,
    #                         position_ids_rmpad=position_ids_rmpad,
    #                         sp_size=self.ulysses_sequence_parallel_size,
    #                     )
    #                 input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
    #                     input_ids_rmpad_rolled,
    #                     position_ids_rmpad=None,
    #                     sp_size=self.ulysses_sequence_parallel_size,
    #                 )

    #             input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

    #             # only pass input_ids and position_ids to enable flash_attn_varlen
    #             extra_args = {}
    #             if self.use_fused_kernels:
    #                 extra_args["temperature"] = temperature

    #             output = self.actor_module(
    #                 input_ids=input_ids_rmpad,
    #                 attention_mask=None,
    #                 position_ids=position_ids_rmpad,
    #                 **multi_modal_inputs,
    #                 use_cache=False,
    #                 **extra_args,
    #             )  # prevent model thinks we are generating

    #             if self.use_fused_kernels:
    #                 log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
    #                 entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

    #             else:
    #                 logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
    #                 logits_rmpad.div_(temperature)

    #                 # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
    #                 inplace_backward = True
    #                 if calculate_entropy:
    #                     inplace_backward = False
    #                 log_probs = logprobs_from_logits(
    #                     logits=logits_rmpad,
    #                     labels=input_ids_rmpad_rolled,
    #                     inplace_backward=inplace_backward,
    #                 )

    #                 # compute entropy
    #                 if calculate_entropy:
    #                     entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

    #             # gather log_prob if sp > 1
    #             if self.use_ulysses_sp:
    #                 # gather and unpad for the ulysses sp
    #                 log_probs = gather_outpus_and_unpad(
    #                     log_probs,
    #                     gather_dim=0,
    #                     unpad_dim=0,
    #                     padding_size=pad_size,
    #                 )
    #                 if calculate_entropy:
    #                     entropy_rmpad = gather_outpus_and_unpad(
    #                         entropy_rmpad,
    #                         gather_dim=0,
    #                         unpad_dim=0,
    #                         padding_size=pad_size,
    #                     )
    #             # pad back to (bsz, seqlen)
    #             if calculate_entropy:
    #                 full_entropy = pad_input(
    #                     hidden_states=entropy_rmpad.unsqueeze(-1),
    #                     indices=indices,
    #                     batch=batch_size,
    #                     seqlen=seqlen,
    #                 )
    #             full_log_probs = pad_input(
    #                 hidden_states=log_probs.unsqueeze(-1),
    #                 indices=indices,
    #                 batch=batch_size,
    #                 seqlen=seqlen,
    #             )

    #             # only return response part:
    #             if calculate_entropy:
    #                 entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
    #             log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

    #         else:  # not using rmpad and no ulysses sp
    #             extra_args = {}
    #             if self.use_fused_kernels:
    #                 extra_args["temperature"] = temperature
    #             output = self.actor_module(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 position_ids=position_ids,
    #                 **multi_modal_inputs,
    #                 use_cache=False,
    #                 **extra_args,
    #             )  # prevent model thinks we are generating

    #             if self.use_fused_kernels:
    #                 log_probs = output.log_probs[:, -response_length - 1 : -1]
    #                 entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

    #             else:
    #                 logits = output.logits

    #                 logits.div_(temperature)
    #                 logits_for_logprob = logits[:, -response_length - 1 : -1, :] # (bsz, response_length, vocab_size)
    #                 log_probs = logprobs_from_logits(logits_for_logprob, micro_batch["responses"])
    #                 if calculate_entropy:
    #                     entropy = verl_F.entropy_from_logits(logits_for_logprob)  # (bsz, response_length)

    #         if return_logits:
    #             return entropy, log_probs, logits_for_logprob
    #         else:
    #             return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            # if torch.distributed.get_rank() == 0:
            #     print("\n" + "="*50)
            #     print("DEBUG INFO: Checking optimizer state devices before step()")
            #     for group in self.actor_optimizer.param_groups:
            #         for p in group['params']:
            #             if p.grad is not None:
            #                 state = self.actor_optimizer.state[p]
            #                 if state: # 如果 state 不为空
            #                     # 检查 exp_avg 和 exp_avg_sq 的设备
            #                     if 'exp_avg' in state and state['exp_avg'].device.type != 'cuda':
            #                         print(f"  [PROBLEM] Param on {p.device}, but exp_avg on {state['exp_avg'].device}")
            #                     if 'exp_avg_sq' in state and state['exp_avg_sq'].device.type != 'cuda':
            #                         print(f"  [PROBLEM] Param on {p.device}, but exp_avg_sq on {state['exp_avg_sq'].device}")
            #     print("="*50 + "\n")
            # for state in self.actor_optimizer.state.values():
            #     for k, v in state.items():
            #         if isinstance(v, torch.Tensor):
            #             state[k] = v.cuda()
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, _ = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    # @GPUMemoryLogger(role="dp actor", logger=logger)
    # def update_policy(self, data: DataProto):
    #     # make sure we are in training mode
    #     self.actor_module.train()

    #     temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
    #     multi_turn = data.meta_info.get("multi_turn", False)

    #     select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
    #     select_keys.extend(["sft_target_responses", "sft_target_mask", "sft_loss_mask"])
    #     if multi_turn:
    #         select_keys.append("loss_mask")
    #     if self.config.use_kl_loss:
    #         select_keys.append("ref_log_prob")
    #     batch = data.select(batch_keys=select_keys).batch
    #     has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

    #     # Split to make minibatch iterator for updating the actor
    #     # See PPO paper for details. https://arxiv.org/abs/1707.06347
    #     if has_multi_modal_inputs:
    #         num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
    #         non_tensor_select_keys = ["multi_modal_inputs"]
    #         dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
    #     else:
    #         dataloader = batch.split(self.config.ppo_mini_batch_size)

    #     metrics = {}
    #     for epoch in range(self.config.ppo_epochs):
    #         for batch_idx, data in enumerate(dataloader):
    #             # split batch into micro_batches
    #             mini_batch = data
    #             if has_multi_modal_inputs:
    #                 self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
    #                 num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
    #                 micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
    #             elif self.config.use_dynamic_bsz:
    #                 max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
    #                 micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
    #             else:
    #                 self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
    #                 # split batch into micro_batches
    #                 micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

    #             self.actor_optimizer.zero_grad()

    #             for data in micro_batches:
    #                 # Support all hardwares
    #                 if isinstance(data, DataProto):
    #                     data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
    #                 else:
    #                     data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
    #                 responses = data["responses"]
    #                 response_length = responses.size(1)
    #                 attention_mask = data["attention_mask"]
    #                 if multi_turn:
    #                     response_mask = data["loss_mask"][:, -response_length:]
    #                 else:
    #                     response_mask = attention_mask[:, -response_length:]

    #                 old_log_prob = data["old_log_probs"]
    #                 advantages = data["advantages"]

    #                 clip_ratio = self.config.clip_ratio
    #                 clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
    #                 clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
    #                 clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
    #                 entropy_coeff = self.config.entropy_coeff
    #                 loss_agg_mode = self.config.loss_agg_mode

    #                 # all return: (bsz, response_length)
    #                 calculate_entropy = False
    #                 if entropy_coeff != 0:
    #                     calculate_entropy = True
    #                 entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

    #                 #=======================================================================#
    #                 # SFT Loss Calculation                                                  #
    #                 #=======================================================================#
                    
    #                 # 1. Get SFT targets if they exist
    #                 sft_target_responses = data.get("sft_target_responses")
    #                 sft_target_mask = data.get("sft_target_mask")
    #                 sft_loss_mask = data.get("sft_loss_mask") # This is per-sample mask
                    
    #                 has_sft_targets = sft_target_responses is not None

    #                 # 2. Get logits from forward pass
    #                 calculate_entropy = self.config.entropy_coeff != 0
    #                 # Request logits for SFT calculation
    #                 forward_output = self._forward_micro_batch(
    #                     micro_batch=data, 
    #                     temperature=temperature, 
    #                     calculate_entropy=calculate_entropy,
    #                     return_logits=has_sft_targets # only get logits if we need them
    #                 )
    #                 if has_sft_targets:
    #                     entropy, log_prob, logits = forward_output
    #                 else:
    #                     entropy, log_prob = forward_output

    #                 pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
    #                     old_log_prob=old_log_prob,
    #                     log_prob=log_prob,
    #                     advantages=advantages,
    #                     response_mask=response_mask,
    #                     cliprange=clip_ratio,
    #                     cliprange_low=clip_ratio_low,
    #                     cliprange_high=clip_ratio_high,
    #                     clip_ratio_c=clip_ratio_c,
    #                     loss_agg_mode=loss_agg_mode,
    #                 )
                    
    #                 # 3. Calculate SFT Loss
    #                 sft_loss = torch.tensor(0.0, device=log_prob.device)
    #                 if has_sft_targets:
    #                     vocab_size = logits.shape[-1]
    #                     sft_logits_fp32 = logits.float()
                        
    #                     # Calculate cross-entropy loss
    #                     loss_fn = nn.CrossEntropyLoss(reduction='none')
    #                     # logits shape: (bs, seq_len, vocab_size) -> (bs * seq_len, vocab_size)
    #                     # target shape: (bs, seq_len) -> (bs * seq_len)
    #                     sft_loss_per_token = loss_fn(
    #                         sft_logits_fp32.reshape(-1, vocab_size),
    #                         sft_target_responses.reshape(-1) 
    #                     ).reshape(responses.shape) #  reshape back to (bs, seq_len)

    #                     # Apply masks
    #                     # sft_target_mask is for padding within the target response
    #                     # sft_loss_mask is to select which samples in the batch have a valid target
    #                     sft_loss_per_sample = masked_mean(sft_loss_per_token, mask=sft_target_mask, axis=-1)
                        
    #                     # Only average over samples that have a valid target
    #                     sft_loss = sft_loss_per_sample[sft_loss_mask].mean()
                        
    #                     # Handle case where no samples in micro-batch have a valid target
    #                     if torch.isnan(sft_loss):
    #                         sft_loss = torch.tensor(0.0, device=log_prob.device)
                        
    #                     # Log the metric
    #                     metrics["actor/sft_loss"] = sft_loss.detach().item()

    #                 #=======================================================================#
    #                 # End SFT Loss Calculation                                              #
    #                 #=======================================================================#


    #                 if entropy_coeff != 0:
    #                     entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    #                     # compute policy loss
    #                     policy_loss = pg_loss - entropy_loss * entropy_coeff
    #                 else:
    #                     policy_loss = pg_loss

    #                 # add SFT loss to policy loss
    #                 # policy_loss += sft_loss * self.config.sft_loss_coeff
    #                 print(f"DEBUG INFO: policy_loss={policy_loss.item()}, pg_loss={pg_loss.item()}, sft_loss={sft_loss.item()}") #, entropy_loss={entropy_loss.item()}

    #                 if self.config.use_kl_loss:
    #                     ref_log_prob = data["ref_log_prob"]
    #                     # compute kl loss
    #                     kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
    #                     kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    #                     policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
    #                     metrics["actor/kl_loss"] = kl_loss.detach().item()
    #                     metrics["actor/kl_coef"] = self.config.kl_loss_coef

    #                 if self.config.use_dynamic_bsz:
    #                     # relative to the dynamic bsz
    #                     loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
    #                 else:
    #                     loss = policy_loss / self.gradient_accumulation
                    
    #                 # if torch.distributed.get_rank() == 0: # 只在 rank 0 打印，避免刷屏
    #                 #     print("\n" + "="*50)
    #                 #     print(f"DEBUG INFO on Rank 0 before backward pass:")
    #                 #     print(f"  - pg_loss device: {pg_loss.device}, dtype: {pg_loss.dtype}")
    #                 #     if 'entropy_loss' in locals():
    #                 #         print(f"  - entropy_loss device: {entropy_loss.device}, dtype: {entropy_loss.dtype}")
    #                 #     if 'sft_loss' in locals():
    #                 #         print(f"  - sft_loss device: {sft_loss.device}, dtype: {sft_loss.dtype}")
    #                 #     if 'policy_loss' in locals():
    #                 #         print(f"  - final policy_loss device: {policy_loss.device}, dtype: {policy_loss.dtype}")
                        
    #                 #     param = next(self.actor_module.parameters())
    #                 #     print(f"  - A model parameter device: {param.device}, dtype: {param.dtype}")
    #                 #     print("="*50 + "\n")
    #                 loss.backward()

    #                 data = {
    #                     "actor/pg_loss": pg_loss.detach().item(),
    #                     "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    #                     "actor/ppo_kl": ppo_kl.detach().item(),
    #                     "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    #                 }
    #                 append_to_dict(metrics, data)

    #             grad_norm = self._optimizer_step()
    #             data = {"actor/grad_norm": grad_norm.detach().item()}
    #             append_to_dict(metrics, data)
    #     self.actor_optimizer.zero_grad()
    #     return metrics
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):

        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        epoch_global = data.meta_info.get("epoch", 0)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        select_keys.extend(["sft_target_responses", "sft_target_mask", "sft_loss_mask"])
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(batch=batch_tensordict_for_rearrange, max_token_len=max_token_len)
                        for current_original_indices, text_mb_td in zip(textual_indices, rearranged_text_micro_batches_tds):
                            current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch_data in micro_batches:
                    if isinstance(micro_batch_data, DataProto):
                        micro_batch_data = {**micro_batch_data.batch.to(get_torch_device().current_device()), **micro_batch_data.non_tensor_batch}
                    elif isinstance(micro_batch_data, dict):
                        for k, v in micro_batch_data.items():
                            if isinstance(v, torch.Tensor):
                                micro_batch_data[k] = v.to(get_torch_device().current_device())
                            elif k == "multi_modal_inputs" and v is not None:
                                micro_batch_data[k] = [{kk: vv.to(get_torch_device().current_device()) for kk, vv in item_dict.items()} for item_dict in v]
                            else:
                                micro_batch_data[k] = v
                    else:
                        micro_batch_data = micro_batch_data.to(get_torch_device().current_device())

                    responses = micro_batch_data["responses"]
                    response_length = responses.size(1)
                    attention_mask = micro_batch_data["attention_mask"]
                    if multi_turn:
                        response_mask = micro_batch_data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = micro_batch_data["old_log_probs"]
                    advantages = micro_batch_data["advantages"]
                    
                    # --- Forward Pass ---
                    entropy, log_prob, logits = self._forward_micro_batch(micro_batch=micro_batch_data, temperature=temperature)

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)

                    # --- PPO Loss ---
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob, 
                        log_prob=log_prob, 
                        advantages=advantages,
                        response_mask=response_mask, 
                        cliprange=self.config.clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )

                    # pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                    #     old_log_prob=old_log_prob,
                    #     log_prob=log_prob,
                    #     advantages=advantages,
                    #     response_mask=response_mask,
                    #     cliprange=clip_ratio,
                    #     cliprange_low=clip_ratio_low,
                    #     cliprange_high=clip_ratio_high,
                    #     clip_ratio_c=clip_ratio_c,
                    #     loss_agg_mode=loss_agg_mode,
                    # )
                    
                    policy_loss = pg_loss

                    # --- Entropy Loss ---
                    if self.config.entropy_coeff != 0 and entropy is not None:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                        policy_loss = policy_loss - entropy_loss * self.config.entropy_coeff

                    # --- SFT Loss ---
                    sft_loss_coeff = self.config.get("sft_loss_coeff", 0.0)
                    sft_start_epoch = self.config.get("sft_start_epoch", 0)
                    # Enable SFT loss only if coeff > 0 AND current epoch >= sft_start_epoch
                    sft_enable = sft_loss_coeff > 0.0 and epoch_global >= sft_start_epoch
                    if sft_loss_coeff > 0.0:
                        sft_target_responses = micro_batch_data.get("sft_target_responses")
                        sft_target_mask = micro_batch_data.get("sft_target_mask")
                        sft_loss_mask = micro_batch_data.get("sft_loss_mask")
                        
                        if sft_target_responses is not None and sft_loss_mask.any():
                            vocab_size = logits.shape[-1]
                            loss_fn = nn.CrossEntropyLoss(reduction='none')
                            
                            sft_logits_fp32 = logits.float() # 强制类型转换
                            sft_loss_per_token = loss_fn(
                                sft_logits_fp32.reshape(-1, vocab_size),
                                sft_target_responses.reshape(-1)
                            ).reshape(responses.shape)
                            
                            sft_loss_per_sample = masked_mean(sft_loss_per_token, mask=sft_target_mask, axis=-1)
                            sft_loss = sft_loss_per_sample[sft_loss_mask].mean()
                            
                            if not torch.isnan(sft_loss):
                                if sft_enable:
                                    policy_loss += sft_loss * sft_loss_coeff
                                    metrics["actor/sft_loss"] = sft_loss.detach().item()
                                else:
                                    metrics["actor/sft_loss"] = -1.0

                    # --- KL Loss ---
                    if self.config.use_kl_loss:
                        ref_log_prob = micro_batch_data["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # --- Backward Pass ---
                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(micro_batch_data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {"actor/pg_loss": pg_loss.detach().item(), "actor/pg_clipfrac": pg_clipfrac.detach().item(), "actor/ppo_kl": ppo_kl.detach().item(), "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
                
        self.actor_optimizer.zero_grad()
        return metrics
