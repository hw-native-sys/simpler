# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 FLASH full 43-layer decode network on 2 dies (EP2 / TP2).

The complete decode forward harvested from pypto-lib
``models/deepseek_v4_flash_mtp/decode_fwd.py``: 43 hand-unrolled layers
(2 SWA + 21 CSA + 20 HCA sparse attentions, one MoE per layer, hash routing,
hyper-connection stack), then ``hc_head -> rms_norm -> lm_head_with_sampling``
TP-sharded over the vocabulary. Attention is data-parallel (each rank decodes
its own micro-batch), the MoE experts are expert-parallel across the two
ranks through the comm window (TPUT payload push + TNOTIFY arrivals), and the
LM head all-gathers hidden states / logits through dedicated windows.

Regime (fixed at harvest time): batch 4 x 2 token rows per rank (T=8),
start_pos 8192, paged KV in 128-token blocks, W8A8 INT8 experts, 64 routed
experts global / 32 per rank at EP2, top-6 + 1 shared.

This is a completion/smoke case (``skip_golden``): upstream pypto-lib runs the
same fixture with ``golden_fn=None`` (component-level golden checks live with
the standalone kernels there); a full-network torch reference does not exist
in either repo. The case validates that the harvested distributed program —
368 incore kernels plus a 7.8k-line chip orchestration per rank — compiles,
dispatches across both dies, drives the comm-window protocol to completion,
and terminates cleanly.

The case participates in the default Per-PR collection. To run it explicitly:

    python examples/a2a3/tensormap_and_ringbuffer/deepseek_v4_flash_decode/\
test_deepseek_v4_flash_decode.py -p a2a3 -d <d0>,<d1>

See README.md for provenance pins and the regeneration recipe.
"""

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.goldens.deepseek_v4_flash_decode import N_RANKS, generate_inputs
from simpler_setup.scene_test import _rehosted_ref

_DIRS = {"i": D.IN, "o": D.OUT, "x": D.INOUT}


def _sig(encoded):
    """Decode a compact direction string (i=IN, o=OUT, x=INOUT) to a signature list."""
    return [_DIRS[c] for c in encoded]


# Incore kernels transcribed from the harvest's kernel_config.py:
# (func_id, name, core_type, signature); source = kernels/<core_type>/<name>.cpp.
_KERNELS = [
    (0, "decode_build_swa_metadata", "aiv", "iix"),
    (1, "decode_build_swa_scalar_metadata", "aiv", "iioo"),
    (2, "decode_build_cache_metadata", "aiv", "iioiioiooioioio"),
    (3, "pack_x_hc", "aiv", "oii"),
    (4, "hc_pre_rms", "aiv", "ix"),
    (5, "hc_pre_linear", "aic", "iix"),
    (6, "hc_pre_linear_reduce", "aiv", "ix"),
    (7, "split_pre_post", "aiv", "iiixxi"),
    (8, "comb_sinkhorn", "aiv", "iiixi"),
    (9, "mix_x", "aiv", "ioi"),
    (10, "swa_rope_step", "aiv", "ooiii"),
    (11, "rms_norm", "aiv", "ioi"),
    (12, "q_rope_prepare", "aiv", "iixxx"),
    (13, "qr_proj_matmul", "aic", "oii"),
    (14, "qr_proj_reduce", "aiv", "ix"),
    (15, "qr_rms_norm_quant", "aiv", "iixoo"),
    (16, "qproj_matmul", "aic", "oii"),
    (17, "qproj_dequant_rms_nope_rope", "aiv", "oiiiiii"),
    (18, "kv_proj_matmul", "aic", "oii"),
    (19, "kv_proj_reduce", "aiv", "ix"),
    (20, "kv_rms_norm_rope", "aiv", "ioiiii"),
    (21, "swa_cache_insert_valid_bias", "aiv", "oiiix"),
    (22, "swa_gather_kv", "aiv", "oii"),
    (23, "qk_pv_aic", "aic", "iioooio"),
    (24, "qk_pv_aiv", "aiv", "iioooio"),
    (25, "rope_cs", "aiv", "xooii"),
    (26, "merge_norm", "aiv", "iiiiiiix"),
    (27, "proj_a_mm", "aic", "iio"),
    (28, "quant", "aiv", "ooi"),
    (29, "proj_b_mm", "aic", "oii"),
    (30, "proj_b_act", "aiv", "iiix"),
    (31, "hc_post", "aiv", "oiiii"),
    (32, "hc_pre_rms_0", "aiv", "ix"),
    (33, "hc_pre_linear_0", "aic", "iix"),
    (34, "hc_pre_linear_reduce_0", "aiv", "ix"),
    (35, "split_pre_post_0", "aiv", "iiixxi"),
    (36, "comb_sinkhorn_0", "aiv", "iiixi"),
    (37, "mix_x_0", "aiv", "ioi"),
    (38, "ffn_norm", "aiv", "iixxxx"),
    (39, "x_norm_quant", "aiv", "ioi"),
    (40, "gate_pre_route", "aiv", "xoox"),
    (41, "gate_aic", "aic", "iiiixo"),
    (42, "gate_aiv", "aiv", "iiiixo"),
    (43, "route_hash", "aiv", "iiixx"),
    (44, "sh_gate_mm", "aic", "iix"),
    (45, "sh_up_mm", "aic", "iix"),
    (46, "sh_gate_up_act_q", "aiv", "ixiiiixo"),
    (47, "sh_w2_mm", "aic", "iix"),
    (48, "sh_w2_act", "aiv", "ioii"),
    (49, "dispatch_meta", "aiv", "ixioo"),
    (50, "dispatch_push", "aiv", "ioiiiooi"),
    (51, "dispatch_wait", "aiv", "ii"),
    (52, "dispatch_gather", "aiv", "oiiiooio"),
    (53, "exp_gate_mm", "aic", "oii"),
    (54, "exp_up_mm", "aic", "oii"),
    (55, "exp_gate_up_act", "aiv", "oiiiiii"),
    (56, "exp_h_q", "aiv", "ixo"),
    (57, "exp_w2_mm", "aic", "oii"),
    (58, "exp_w2_act", "aiv", "iioii"),
    (59, "combine", "aiv", "iioi"),
    (60, "combine_wait", "aiv", "i"),
    (61, "shared_routed", "aiv", "iix"),
    (62, "hc_post_0", "aiv", "oiiii"),
    (63, "hc_pre_rms_1", "aiv", "ix"),
    (64, "hc_pre_linear_1", "aic", "iix"),
    (65, "hc_pre_linear_reduce_1", "aiv", "ix"),
    (66, "split_pre_post_1", "aiv", "iiixxi"),
    (67, "comb_sinkhorn_1", "aiv", "iiixi"),
    (68, "mix_x_1", "aiv", "ioi"),
    (69, "swa_rope_step_0", "aiv", "ooiii"),
    (70, "rms_norm_0", "aiv", "ioi"),
    (71, "q_rope_prepare_0", "aiv", "iixxx"),
    (72, "qr_proj_matmul_0", "aic", "oii"),
    (73, "qr_proj_reduce_0", "aiv", "ix"),
    (74, "qr_rms_norm_quant_0", "aiv", "iixoo"),
    (75, "qproj_matmul_0", "aic", "oii"),
    (76, "qproj_dequant_rms_nope_rope_0", "aiv", "oiiiiii"),
    (77, "kv_proj_matmul_0", "aic", "oii"),
    (78, "kv_proj_reduce_0", "aiv", "ix"),
    (79, "kv_rms_norm_rope_0", "aiv", "ioiiii"),
    (80, "swa_cache_insert_valid_bias_0", "aiv", "oiiix"),
    (81, "swa_gather_kv_0", "aiv", "oii"),
    (82, "qk_pv_0_aic", "aic", "iioooio"),
    (83, "qk_pv_0_aiv", "aiv", "iioooio"),
    (84, "rope_cs_0", "aiv", "xooii"),
    (85, "merge_norm_0", "aiv", "iiiiiiix"),
    (86, "proj_a_mm_0", "aic", "iio"),
    (87, "quant_0", "aiv", "ooi"),
    (88, "proj_b_mm_0", "aic", "oii"),
    (89, "proj_b_act_0", "aiv", "iiix"),
    (90, "hc_post_1", "aiv", "oiiii"),
    (91, "hc_pre_rms_2", "aiv", "ix"),
    (92, "hc_pre_linear_2", "aic", "iix"),
    (93, "hc_pre_linear_reduce_2", "aiv", "ix"),
    (94, "split_pre_post_2", "aiv", "iiixxi"),
    (95, "comb_sinkhorn_2", "aiv", "iiixi"),
    (96, "mix_x_2", "aiv", "ioi"),
    (97, "ffn_norm_0", "aiv", "iixxxx"),
    (98, "x_norm_quant_0", "aiv", "ioi"),
    (99, "gate_pre_route_0", "aiv", "xoox"),
    (100, "gate_0_aic", "aic", "iiiixo"),
    (101, "gate_0_aiv", "aiv", "iiiixo"),
    (102, "route_hash_0", "aiv", "iiixx"),
    (103, "sh_gate_mm_0", "aic", "iix"),
    (104, "sh_up_mm_0", "aic", "iix"),
    (105, "sh_gate_up_act_q_0", "aiv", "ixiiiixo"),
    (106, "sh_w2_mm_0", "aic", "iix"),
    (107, "sh_w2_act_0", "aiv", "ioii"),
    (108, "dispatch_meta_0", "aiv", "ixioo"),
    (109, "dispatch_push_0", "aiv", "ixiiixxi"),
    (110, "dispatch_wait_0", "aiv", "ii"),
    (111, "dispatch_gather_0", "aiv", "oiiiooio"),
    (112, "exp_gate_mm_0", "aic", "oii"),
    (113, "exp_up_mm_0", "aic", "oii"),
    (114, "exp_gate_up_act_0", "aiv", "oiiiiii"),
    (115, "exp_h_q_0", "aiv", "ixo"),
    (116, "exp_w2_mm_0", "aic", "oii"),
    (117, "exp_w2_act_0", "aiv", "iioii"),
    (118, "combine_0", "aiv", "iixi"),
    (119, "combine_wait_0", "aiv", "i"),
    (120, "shared_routed_0", "aiv", "iix"),
    (121, "hc_post_2", "aiv", "oiiii"),
    (122, "hc_pre_rms_3", "aiv", "ix"),
    (123, "hc_pre_linear_3", "aic", "iix"),
    (124, "hc_pre_linear_reduce_3", "aiv", "ix"),
    (125, "split_pre_post_3", "aiv", "iiixxi"),
    (126, "comb_sinkhorn_3", "aiv", "iiixi"),
    (127, "mix_x_3", "aiv", "ixi"),
    (128, "csa_rope_step", "aiv", "xxxxiii"),
    (129, "rope_interleave", "aiv", "ixix"),
    (130, "csa_cmp_rope", "aiv", "xxiii"),
    (131, "rope_interleave_0", "aiv", "ixix"),
    (132, "rms_norm_1", "aiv", "ixi"),
    (133, "q_rope_prepare_1", "aiv", "iixxx"),
    (134, "qr_proj_matmul_1", "aic", "xii"),
    (135, "qr_proj_reduce_1", "aiv", "ix"),
    (136, "qr_rms_norm_quant_1", "aiv", "iixxx"),
    (137, "qproj_matmul_1", "aic", "xii"),
    (138, "qproj_dequant_rms_nope_rope_1", "aiv", "xiiiiii"),
    (139, "kv_proj_matmul_1", "aic", "xii"),
    (140, "kv_proj_reduce_1", "aiv", "ix"),
    (141, "kv_rms_norm_rope_1", "aiv", "ixiiii"),
    (142, "csa_cache_writeback", "aiv", "xii"),
    (143, "kv_score_proj", "aic", "iiixx"),
    (144, "scatter_softmax_pool", "aiv", "xxiiiiii"),
    (145, "rmsnorm_rope_cache_write", "aiv", "iiixixxii"),
    (146, "idx_qr_proj_matmul", "aic", "xii"),
    (147, "idx_qr_proj_dequant", "aiv", "iiix"),
    (148, "qr_rope_swap_idx", "aiv", "x"),
    (149, "qr_rope", "aiv", "iiiix"),
    (150, "qr_hadamard_matmul", "aic", "iix"),
    (151, "qr_hadamard_quant", "aiv", "ixx"),
    (152, "weights_proj", "aic", "iix"),
    (153, "weights_proj_reduce", "aiv", "ix"),
    (154, "kv_score_proj_0", "aic", "iiixx"),
    (155, "scatter_softmax_pool_0", "aiv", "xxiiiiii"),
    (156, "rmsnorm_rope", "aiv", "iiixi"),
    (157, "kv_hadamard", "aic", "ixi"),
    (158, "kv_and_cache_write", "aiv", "ixxiix"),
    (159, "score_aic", "aic", "iiiiixiiix"),
    (160, "score_aiv", "aiv", "iiiiixiiix"),
    (161, "topk", "aiv", "xiii"),
    (162, "kv_touch", "aiv", "x"),
    (163, "csa_slots_build_valid_qk_plan", "aiv", "iixxixxx"),
    (164, "qk_pv_1_aic", "aic", "xxxiiiiiiiiiix"),
    (165, "qk_pv_1_aiv", "aiv", "xxxiiiiiiiiiix"),
    (166, "rope_cs_1", "aiv", "xiixx"),
    (167, "merge_norm_1", "aiv", "iiiiiiix"),
    (168, "proj_a_mm_1", "aic", "iix"),
    (169, "quant_1", "aiv", "xxi"),
    (170, "proj_b_mm_1", "aic", "xii"),
    (171, "proj_b_act_1", "aiv", "iiix"),
    (172, "hc_post_3", "aiv", "xiiii"),
    (173, "hc_pre_rms_4", "aiv", "ix"),
    (174, "hc_pre_linear_4", "aic", "iix"),
    (175, "hc_pre_linear_reduce_4", "aiv", "ix"),
    (176, "split_pre_post_4", "aiv", "iiixxi"),
    (177, "comb_sinkhorn_4", "aiv", "iiixi"),
    (178, "mix_x_4", "aiv", "ixi"),
    (179, "ffn_norm_1", "aiv", "iixxxx"),
    (180, "x_norm_quant_1", "aiv", "ixi"),
    (181, "gate_pre_route_1", "aiv", "xxxx"),
    (182, "gate_1_aic", "aic", "iiiixxx"),
    (183, "gate_1_aiv", "aiv", "iiiixxx"),
    (184, "route_hash_1", "aiv", "iiixx"),
    (185, "route_sort", "aiv", "iixx"),
    (186, "sh_gate_mm_1", "aic", "iix"),
    (187, "sh_up_mm_1", "aic", "iix"),
    (188, "sh_gate_up_act_q_1", "aiv", "ixiiiixx"),
    (189, "sh_w2_mm_1", "aic", "iix"),
    (190, "sh_w2_act_1", "aiv", "ixii"),
    (191, "dispatch_meta_1", "aiv", "ixixx"),
    (192, "dispatch_push_1", "aiv", "ixiiixxi"),
    (193, "dispatch_wait_1", "aiv", "ii"),
    (194, "dispatch_gather_1", "aiv", "xiiixxix"),
    (195, "exp_gate_mm_1", "aic", "xii"),
    (196, "exp_up_mm_1", "aic", "xii"),
    (197, "exp_gate_up_act_1", "aiv", "xiiiiii"),
    (198, "exp_h_q_1", "aiv", "ixx"),
    (199, "exp_w2_mm_1", "aic", "xii"),
    (200, "exp_w2_act_1", "aiv", "iixii"),
    (201, "combine_1", "aiv", "iixi"),
    (202, "combine_wait_1", "aiv", "i"),
    (203, "shared_routed_1", "aiv", "iix"),
    (204, "hc_post_4", "aiv", "xiiii"),
    (205, "hc_pre_rms_5", "aiv", "ix"),
    (206, "hc_pre_linear_5", "aic", "iix"),
    (207, "hc_pre_linear_reduce_5", "aiv", "ix"),
    (208, "split_pre_post_5", "aiv", "iiixxi"),
    (209, "comb_sinkhorn_5", "aiv", "iiixi"),
    (210, "mix_x_5", "aiv", "ixi"),
    (211, "hca_rope", "aiv", "xxxxiii"),
    (212, "rope_interleave_1", "aiv", "ixix"),
    (213, "rms_norm_2", "aiv", "ixi"),
    (214, "q_rope_prepare_2", "aiv", "iixxx"),
    (215, "qr_proj_matmul_2", "aic", "xii"),
    (216, "qr_proj_reduce_2", "aiv", "ix"),
    (217, "qr_rms_norm_quant_2", "aiv", "iixxx"),
    (218, "qproj_matmul_2", "aic", "xii"),
    (219, "qproj_dequant_rms_nope_rope_2", "aiv", "xiiiiii"),
    (220, "kv_proj_matmul_2", "aic", "xii"),
    (221, "kv_proj_reduce_2", "aiv", "ix"),
    (222, "kv_rms_norm_rope_2", "aiv", "ixiiii"),
    (223, "hca_cache_writeback", "aiv", "xii"),
    (224, "kv_score_proj_1", "aic", "iiixx"),
    (225, "scatter_softmax_pool_1", "aiv", "xxiiiiii"),
    (226, "rmsnorm_rope_cache_write_0", "aiv", "iiixixxii"),
    (227, "hca_cache_topk", "aiv", "iix"),
    (228, "build_valid", "aiv", "iix"),
    (229, "hca_gather_kv", "aiv", "xiiiii"),
    (230, "qk_pv_2_aic", "aic", "iixxxix"),
    (231, "qk_pv_2_aiv", "aiv", "iixxxix"),
    (232, "rope_swap", "aiv", "x"),
    (233, "rope_cs_2", "aiv", "iixx"),
    (234, "merge_norm_2", "aiv", "iiiiiiix"),
    (235, "proj_a_mm_2", "aic", "iix"),
    (236, "quant_2", "aiv", "xxi"),
    (237, "proj_b_mm_2", "aic", "xii"),
    (238, "proj_b_act_2", "aiv", "iiix"),
    (239, "hc_post_5", "aiv", "xiiii"),
    (240, "hc_pre_rms_6", "aiv", "ix"),
    (241, "hc_pre_linear_6", "aic", "iix"),
    (242, "hc_pre_linear_reduce_6", "aiv", "ix"),
    (243, "split_pre_post_6", "aiv", "iiixxi"),
    (244, "comb_sinkhorn_6", "aiv", "iiixi"),
    (245, "mix_x_6", "aiv", "ixi"),
    (246, "ffn_norm_2", "aiv", "iixxxx"),
    (247, "x_norm_quant_2", "aiv", "ixi"),
    (248, "gate_pre_route_2", "aiv", "xxxx"),
    (249, "gate_2_aic", "aic", "iiiixxx"),
    (250, "gate_2_aiv", "aiv", "iiiixxx"),
    (251, "route_sort_0", "aiv", "iixx"),
    (252, "sh_gate_mm_2", "aic", "iix"),
    (253, "sh_up_mm_2", "aic", "iix"),
    (254, "sh_gate_up_act_q_2", "aiv", "ixiiiixx"),
    (255, "sh_w2_mm_2", "aic", "iix"),
    (256, "sh_w2_act_2", "aiv", "ixii"),
    (257, "dispatch_meta_2", "aiv", "ixixx"),
    (258, "dispatch_push_2", "aiv", "ixiiixxi"),
    (259, "dispatch_wait_2", "aiv", "ii"),
    (260, "dispatch_gather_2", "aiv", "xiiixxix"),
    (261, "exp_gate_mm_2", "aic", "xii"),
    (262, "exp_up_mm_2", "aic", "xii"),
    (263, "exp_gate_up_act_2", "aiv", "xiiiiii"),
    (264, "exp_h_q_2", "aiv", "ixx"),
    (265, "exp_w2_mm_2", "aic", "xii"),
    (266, "exp_w2_act_2", "aiv", "iixii"),
    (267, "combine_2", "aiv", "iixi"),
    (268, "combine_wait_2", "aiv", "i"),
    (269, "shared_routed_2", "aiv", "iix"),
    (270, "hc_post_6", "aiv", "xiiii"),
    (271, "hc_pre_rms_7", "aiv", "ix"),
    (272, "hc_pre_linear_7", "aic", "iix"),
    (273, "hc_pre_linear_reduce_7", "aiv", "ix"),
    (274, "split_pre_post_7", "aiv", "iiixxi"),
    (275, "comb_sinkhorn_7", "aiv", "iiixi"),
    (276, "mix_x_7", "aiv", "ioi"),
    (277, "csa_rope_step_0", "aiv", "ooooiii"),
    (278, "rope_interleave_2", "aiv", "ixix"),
    (279, "csa_cmp_rope_0", "aiv", "ooiii"),
    (280, "rope_interleave_3", "aiv", "ixix"),
    (281, "rms_norm_3", "aiv", "ioi"),
    (282, "q_rope_prepare_3", "aiv", "iixxx"),
    (283, "qr_proj_matmul_3", "aic", "oii"),
    (284, "qr_proj_reduce_3", "aiv", "ix"),
    (285, "qr_rms_norm_quant_3", "aiv", "iixoo"),
    (286, "qproj_matmul_3", "aic", "oii"),
    (287, "qproj_dequant_rms_nope_rope_3", "aiv", "oiiiiii"),
    (288, "kv_proj_matmul_3", "aic", "oii"),
    (289, "kv_proj_reduce_3", "aiv", "ix"),
    (290, "kv_rms_norm_rope_3", "aiv", "ioiiii"),
    (291, "csa_cache_writeback_0", "aiv", "oii"),
    (292, "kv_score_proj_2", "aic", "iiixx"),
    (293, "scatter_softmax_pool_2", "aiv", "xoiiiiii"),
    (294, "rmsnorm_rope_cache_write_1", "aiv", "iiixiooii"),
    (295, "idx_qr_proj_matmul_0", "aic", "oii"),
    (296, "idx_qr_proj_dequant_0", "aiv", "iiix"),
    (297, "qr_rope_swap_idx_0", "aiv", "x"),
    (298, "qr_rope_0", "aiv", "iiiix"),
    (299, "qr_hadamard_matmul_0", "aic", "iix"),
    (300, "qr_hadamard_quant_0", "aiv", "ixo"),
    (301, "weights_proj_0", "aic", "iix"),
    (302, "weights_proj_reduce_0", "aiv", "ix"),
    (303, "kv_score_proj_3", "aic", "iiixx"),
    (304, "scatter_softmax_pool_3", "aiv", "xoiiiiii"),
    (305, "rmsnorm_rope_0", "aiv", "iiioi"),
    (306, "kv_hadamard_0", "aic", "ioi"),
    (307, "kv_and_cache_write_0", "aiv", "iooiio"),
    (308, "score_0_aic", "aic", "iiiiioiiio"),
    (309, "score_0_aiv", "aiv", "iiiiioiiio"),
    (310, "topk_0", "aiv", "xiii"),
    (311, "kv_touch_0", "aiv", "x"),
    (312, "csa_slots_build_valid_qk_plan_0", "aiv", "iixxixxo"),
    (313, "qk_pv_3_aic", "aic", "oooiiiiiiiiiio"),
    (314, "qk_pv_3_aiv", "aiv", "oooiiiiiiiiiio"),
    (315, "rope_cs_3", "aiv", "xiixx"),
    (316, "merge_norm_3", "aiv", "iiiiiiix"),
    (317, "proj_a_mm_3", "aic", "iio"),
    (318, "quant_3", "aiv", "ooi"),
    (319, "proj_b_mm_3", "aic", "oii"),
    (320, "proj_b_act_3", "aiv", "iiix"),
    (321, "hc_post_7", "aiv", "oiiii"),
    (322, "hc_pre_rms_8", "aiv", "ix"),
    (323, "hc_pre_linear_8", "aic", "iix"),
    (324, "hc_pre_linear_reduce_8", "aiv", "ix"),
    (325, "split_pre_post_8", "aiv", "iiixxi"),
    (326, "comb_sinkhorn_8", "aiv", "iiixi"),
    (327, "mix_x_8", "aiv", "ioi"),
    (328, "ffn_norm_3", "aiv", "iixxxx"),
    (329, "x_norm_quant_3", "aiv", "ioi"),
    (330, "gate_pre_route_3", "aiv", "xoox"),
    (331, "gate_3_aic", "aic", "iiiixxo"),
    (332, "gate_3_aiv", "aiv", "iiiixxo"),
    (333, "route_sort_1", "aiv", "iixx"),
    (334, "sh_gate_mm_3", "aic", "iix"),
    (335, "sh_up_mm_3", "aic", "iix"),
    (336, "sh_gate_up_act_q_3", "aiv", "ixiiiixo"),
    (337, "sh_w2_mm_3", "aic", "iix"),
    (338, "sh_w2_act_3", "aiv", "ioii"),
    (339, "dispatch_meta_3", "aiv", "ixioo"),
    (340, "dispatch_push_3", "aiv", "ixiiixxi"),
    (341, "dispatch_wait_3", "aiv", "ii"),
    (342, "dispatch_gather_3", "aiv", "oiiiooio"),
    (343, "exp_gate_mm_3", "aic", "oii"),
    (344, "exp_up_mm_3", "aic", "oii"),
    (345, "exp_gate_up_act_3", "aiv", "oiiiiii"),
    (346, "exp_h_q_3", "aiv", "ixo"),
    (347, "exp_w2_mm_3", "aic", "oii"),
    (348, "exp_w2_act_3", "aiv", "iioii"),
    (349, "combine_3", "aiv", "iixi"),
    (350, "combine_wait_3", "aiv", "i"),
    (351, "shared_routed_3", "aiv", "iix"),
    (352, "hc_post_8", "aiv", "oiiii"),
    (353, "moe_signal_clear", "aiv", "iooo"),
    (354, "hc_head_rms", "aiv", "ix"),
    (355, "hc_head_linear", "aic", "iix"),
    (356, "hc_head_reduce", "aiv", "iiiioi"),
    (357, "rms_norm_4", "aiv", "ioi"),
    (358, "lm_head_dispatch_push", "aiv", "iixoi"),
    (359, "lm_head_dispatch_wait", "aiv", "ii"),
    (360, "lm_head_dispatch_gather", "aiv", "ix"),
    (361, "lm_head_matmul", "aic", "oii"),
    (362, "lm_head_combine_push", "aiv", "oii"),
    (363, "lm_head_combine_wait", "aiv", "ii"),
    (364, "lm_head_combine_gather", "aiv", "oi"),
    (365, "lm_head_signal_clear", "aiv", "ioo"),
    (366, "lm_head_greedy_sample", "aiv", "ix"),
    (367, "hc_head_mixes_zero", "aiv", "o"),
]

# Chip orchestration tensor-argument directions (scalars excluded), from kernel_config.py.
_ORCH_SIG = "iiiiiiiiiiixiiiiiiiixiiiixiiiiiiiixxxxiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiioooxiiiiooioooooi"

# Comm-window buffers from the generated host orchestration (dtype opaque, 32 B aligned).
_WINDOW_BUFFERS = [
    ("recv_meta_buf", 256),
    ("recv_x_buf", 2097152),
    ("recv_aux_buf", 16384),
    ("recv_route_buf", 16384),
    ("arrived_buf", 8),
    ("data_arrived_buf", 8),
    ("routed_y_buf_buf", 393216),
    ("combine_arrived_buf", 8),
    ("lm_head_hidden_window_buf", 131072),
    ("lm_head_logits_window_buf", 4136960),
    ("lm_head_hidden_done_buf", 8),
    ("lm_head_logits_done_buf", 8),
]

_WINDOW_SIZE = sum(-(-count // 32) * 32 for _, count in _WINDOW_BUFFERS)

# Per-rank TaskArgs build steps, transcribed from the generated host_orch.py:
#   ("slice",  name, argtype)               -> this rank's shard  (host tensor  f"{name}_r{rank}")
#   ("window", name, shape, dtype, argtype) -> comm-window view   (domain buffer tensor)
#   ("whole",  name, argtype)               -> whole host tensor  (all ranks' values)
_ARG_STEPS: list[tuple] = [
    ("slice", "embed_weight", "INPUT"),
    ("slice", "hc_attn_fn", "INPUT"),
    ("slice", "hc_attn_scale", "INPUT"),
    ("slice", "hc_attn_base", "INPUT"),
    ("slice", "attn_norm_w", "INPUT"),
    ("slice", "wq_a", "INPUT"),
    ("slice", "wq_b", "INPUT"),
    ("slice", "wq_b_scale", "INPUT"),
    ("slice", "wkv", "INPUT"),
    ("slice", "gamma_cq", "INPUT"),
    ("slice", "gamma_ckv", "INPUT"),
    ("slice", "kv_cache", "INOUT"),
    ("slice", "attn_sink", "INPUT"),
    ("slice", "wo_a", "INPUT"),
    ("slice", "wo_b", "INPUT"),
    ("slice", "wo_b_scale", "INPUT"),
    ("slice", "hca_cmp_wkv", "INPUT"),
    ("slice", "hca_cmp_wgate", "INPUT"),
    ("slice", "hca_cmp_ape", "INPUT"),
    ("slice", "hca_cmp_norm_w", "INPUT"),
    ("slice", "hca_compress_state", "INOUT"),
    ("slice", "csa_cmp_wkv", "INPUT"),
    ("slice", "csa_cmp_wgate", "INPUT"),
    ("slice", "csa_cmp_ape", "INPUT"),
    ("slice", "csa_cmp_norm_w", "INPUT"),
    ("slice", "csa_compress_state", "INOUT"),
    ("slice", "csa_idx_wq_b", "INPUT"),
    ("slice", "csa_idx_wq_b_scale", "INPUT"),
    ("slice", "csa_weights_proj", "INPUT"),
    ("slice", "csa_hadamard_idx", "INPUT"),
    ("slice", "csa_inner_wkv", "INPUT"),
    ("slice", "csa_inner_wgate", "INPUT"),
    ("slice", "csa_inner_ape", "INPUT"),
    ("slice", "csa_inner_norm_w", "INPUT"),
    ("slice", "csa_inner_compress_state", "INOUT"),
    ("slice", "cmp_kv", "INOUT"),
    ("slice", "idx_kv_cache", "INOUT"),
    ("slice", "idx_kv_scale", "INOUT"),
    ("slice", "hc_ffn_fn", "INPUT"),
    ("slice", "hc_ffn_scale", "INPUT"),
    ("slice", "hc_ffn_base", "INPUT"),
    ("slice", "norm_w", "INPUT"),
    ("slice", "gate_w", "INPUT"),
    ("slice", "gate_bias", "INPUT"),
    ("slice", "tid2eid", "INPUT"),
    ("slice", "routed_w1", "INPUT"),
    ("slice", "routed_w1_scale", "INPUT"),
    ("slice", "routed_w3", "INPUT"),
    ("slice", "routed_w3_scale", "INPUT"),
    ("slice", "routed_w2", "INPUT"),
    ("slice", "routed_w2_scale", "INPUT"),
    ("slice", "shared_w1", "INPUT"),
    ("slice", "shared_w1_scale", "INPUT"),
    ("slice", "shared_w3", "INPUT"),
    ("slice", "shared_w3_scale", "INPUT"),
    ("slice", "shared_w2", "INPUT"),
    ("slice", "shared_w2_scale", "INPUT"),
    ("slice", "freqs_cos", "INPUT"),
    ("slice", "freqs_sin", "INPUT"),
    ("slice", "block_table", "INPUT"),
    ("slice", "position_ids", "INPUT"),
    ("slice", "kv_seq_lens", "INPUT"),
    ("slice", "hca_compress_state_block_table", "INPUT"),
    ("slice", "csa_compress_state_block_table", "INPUT"),
    ("slice", "csa_inner_compress_state_block_table", "INPUT"),
    ("slice", "cmp_block_table", "INPUT"),
    ("slice", "idx_block_table", "INPUT"),
    ("slice", "block_counts", "INPUT"),
    ("slice", "input_ids", "INPUT"),
    ("slice", "hc_head_fn", "INPUT"),
    ("slice", "hc_head_scale", "INPUT"),
    ("slice", "hc_head_base", "INPUT"),
    ("slice", "final_norm_w", "INPUT"),
    ("slice", "lm_head_weight", "INPUT"),
    ("slice", "logit_row_indices", "INPUT"),
    ("slice", "pre_hc_hidden_out", "OUTPUT_EXISTING"),
    ("slice", "hidden_out", "OUTPUT_EXISTING"),
    ("slice", "logits", "OUTPUT_EXISTING"),
    ("slice", "sampled_ids", "INOUT"),
    ("window", "recv_meta_buf", (2, 32), "INT32", "INPUT"),
    ("window", "recv_x_buf", (512, 4096), "INT8", "INPUT"),
    ("window", "recv_aux_buf", (512, 8), "FLOAT32", "INPUT"),
    ("window", "recv_route_buf", (512, 8), "INT32", "INPUT"),
    ("window", "arrived_buf", (2, 1), "INT32", "OUTPUT_EXISTING"),
    ("window", "data_arrived_buf", (2, 1), "INT32", "OUTPUT_EXISTING"),
    ("window", "routed_y_buf_buf", (48, 4096), "BFLOAT16", "INPUT"),
    ("window", "combine_arrived_buf", (2, 1), "INT32", "OUTPUT_EXISTING"),
    ("window", "lm_head_hidden_window_buf", (16, 4096), "BFLOAT16", "OUTPUT_EXISTING"),
    ("window", "lm_head_hidden_done_buf", (2, 1), "INT32", "OUTPUT_EXISTING"),
    ("window", "lm_head_logits_window_buf", (8, 129280), "FLOAT32", "OUTPUT_EXISTING"),
    ("window", "lm_head_logits_done_buf", (2, 1), "INT32", "OUTPUT_EXISTING"),
    ("whole", "num_tokens_per_owner", "INPUT"),
]

_N_CTX_SCALARS = 12


def _decode_fwd_orch_fn(orch, callables, task_args, config):
    with orch.allocate_domain(
        name="comm_d0",
        workers=list(range(N_RANKS)),
        window_size=_WINDOW_SIZE,
        buffers=[
            CommBufferSpec(name=name, dtype="opaque", count=count, nbytes=-(-count // 32) * 32)
            for name, count in _WINDOW_BUFFERS
        ],
    ) as domain:
        for rank in range(N_RANKS):
            dom = domain[rank]
            args = TaskArgs()
            for step in _ARG_STEPS:
                arg_type = getattr(TensorArgType, step[-1])
                if step[0] == "slice":
                    args.add_tensor(_rehosted_ref(task_args, f"{step[1]}_r{rank}"), arg_type)
                elif step[0] == "window":
                    args.add_tensor(
                        dom.buffers[step[1]].tensor(step[2], getattr(DataType, step[3])),
                        arg_type,
                    )
                else:
                    args.add_tensor(_rehosted_ref(task_args, step[1]), arg_type)
            args.add_scalar(rank)
            for _ in range(_N_CTX_SCALARS):
                args.add_scalar(dom.device_ctx)
            callables.keep(args)
            orch.submit_next_level(callables.decode_fwd, args, config, worker=rank)


@pytest.mark.resource_last
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestDeepseekV4FlashDecode(SceneTestCase):
    CALLABLE = {
        "orchestration": _decode_fwd_orch_fn,
        "callables": [
            {
                "name": "decode_fwd",
                "orchestration": {
                    "source": "kernels/orchestration/decode_fwd.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": _sig(_ORCH_SIG),
                },
                "incores": [
                    {
                        "func_id": func_id,
                        "name": name,
                        "source": f"kernels/{core_type}/{name}.cpp",
                        "core_type": core_type,
                        "signature": _sig(sig),
                    }
                    for func_id, name, core_type, sig in _KERNELS
                ],
            }
        ],
    }
    CASES = [
        {
            "name": "DecodeFwdEP2TP2",
            "platforms": ["a2a3"],
            "skip_golden": True,
            "config": {
                "device_count": N_RANKS,
                "num_sub_workers": 0,
                # Ring sizing for the 43-layer graph. The task window and dep
                # pool match pypto-lib's daily CI env for this network
                # (PTO2_RING_* 16384 / 16384). The heap is twice that env's
                # 1 GiB: the MoE per-expert tile grid is static, so tile scratch
                # is allocated for all 32 experts of every MoE layer whatever the
                # routing turns out to be.
                "runtime_env": {
                    "ring_task_window": 16384,
                    "ring_heap": 2 << 30,
                    "ring_dep_pool": 16384,
                },
            },
            "params": {"seed": 1234},
        }
    ]

    def generate_args(self, params):
        return generate_inputs(params.get("seed", 1234))

    def compute_golden(self, args, params):
        raise NotImplementedError(
            "deepseek_v4_flash_decode is a completion/smoke case (skip_golden): no "
            "full-network torch reference exists upstream either. Component-level "
            "goldens live with the standalone kernels in pypto-lib."
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
