    Tensor pre_mix = pre_mix__rv_v1;
    Tensor token_cos = compressed_token_rope_cos;
    Tensor token_sin = compressed_token_rope_sin;
    uint32_t indices_attention_count_inline23434[1] = {static_cast<uint32_t>(rank)};
    int32_t attention_count_inline23434 = get_tensor_data<int32_t>(ext_attention_num_tokens, 1, indices_attention_count_inline23434);
    uint32_t compressed_token_rope_cos_inline23437_offsets[2] = {0, 0};
    uint32_t compressed_token_rope_cos_inline23437_shapes[2] = {(compressed_token_rope_cos_inline23437_offsets[0] >= token_cos.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(T_DYN), token_cos.shapes[0] - compressed_token_rope_cos_inline23437_offsets[0])), (compressed_token_rope_cos_inline23437_offsets[1] >= token_cos.shapes[1] ? 0u : std::min<uint32_t>(32, token_cos.shapes[1] - compressed_token_rope_cos_inline23437_offsets[1]))};
    Tensor compressed_token_rope_cos_inline23437 = token_cos.view(compressed_token_rope_cos_inline23437_shapes, compressed_token_rope_cos_inline23437_offsets);
    uint32_t compressed_token_rope_sin_inline23438_offsets[2] = {0, 0};
    uint32_t compressed_token_rope_sin_inline23438_shapes[2] = {(compressed_token_rope_sin_inline23438_offsets[0] >= token_sin.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(T_DYN), token_sin.shapes[0] - compressed_token_rope_sin_inline23438_offsets[0])), (compressed_token_rope_sin_inline23438_offsets[1] >= token_sin.shapes[1] ? 0u : std::min<uint32_t>(32, token_sin.shapes[1] - compressed_token_rope_sin_inline23438_offsets[1]))};
    Tensor compressed_token_rope_sin_inline23438 = token_sin.view(compressed_token_rope_sin_inline23438_shapes, compressed_token_rope_sin_inline23438_offsets);
    uint32_t positions_row_inline23511_shapes[2] = {1, static_cast<uint32_t>(T_DYN)};
    Tensor positions_row_inline23511 = ext_position_ids.reshape(positions_row_inline23511_shapes, 2);
    uint32_t lengths_row_inline23512_shapes[2] = {1, static_cast<uint32_t>(T_DYN)};
    Tensor lengths_row_inline23512 = c1a_compressed_lens_inline23439.reshape(lengths_row_inline23512_shapes, 2);
    for (int64_t row_inline23513 = 0; row_inline23513 < T_DYN; row_inline23513 += 128) {

        // Task 131: prepare_compressed_lens_0
        CoreTaskArgs params_t131;
        params_t131.add_input(positions_row_inline23511);
        params_t131.add_output(lengths_row_inline23512);
        params_t131.add_scalar(T_DYN);
        params_t131.add_scalar(row_inline23513);
        rt_submit_aiv_task(93, params_t131);
    }
    int64_t window_blocks_per_layer_inline23440 = (FWD_ORI_BLOCKS_DYN / 40);
    int64_t compressed_blocks_per_source_inline23441 = (FWD_CMP_BLOCKS_DYN / 4);
    int64_t index_blocks_per_source_inline23442 = (FWD_INDEX_BLOCKS_DYN / 4);
    int64_t pool_blocks_raw_inline23446 = (FWD_C1A_TABLE_DYN * 16);
    int64_t pool_block_capacity_inline23447 = std::min<int64_t>(std::max<int64_t>((((pool_blocks_raw_inline23446 + 31) / 32) * 32), 32), 2048);
    uint32_t pool_block_ids_inline23448_ci_shapes[2] = {static_cast<uint32_t>(T_DYN), static_cast<uint32_t>(pool_block_capacity_inline23447)};
    TensorCreateInfo pool_block_ids_inline23448_ci(pool_block_ids_inline23448_ci_shapes, 2, DataType::INT32);
    TaskOutputTensors alloc_51 = alloc_tensors(pool_block_ids_inline23448_ci);
    const Tensor& pool_block_ids_inline23448 = alloc_51.get_ref(0);
    uint32_t indices_local_moe_tokens_inline23451[1] = {static_cast<uint32_t>(rank)};
    int32_t local_moe_tokens_inline23451 = get_tensor_data<int32_t>(ext_moe_num_tokens, 1, indices_local_moe_tokens_inline23451);
    Tensor attention_pre_mix_inline23444__rv_v3 = attention_pre_mix_inline23444;
    Tensor topk_indices_inline23445__rv_v5 = topk_indices_inline23445;
    Tensor pool_block_ids_inline23448__rv_v3 = pool_block_ids_inline23448;
    Tensor encoder_pre_mix__rv_v1 = ext_pre_mix;
    Tensor x_mixed_inline23450__rv_v1 = x_mixed_inline23450;
    for (int64_t layer_inline23452 = 20; layer_inline23452 < 40; layer_inline23452 += 1) {
        Tensor attention_pre_mix_inline23444__rv_v2 = ext_engram_lookup;
        Tensor topk_indices_inline23445__rv_v4 = ext_engram_lookup;
        Tensor pool_block_ids_inline23448__rv_v2 = ext_engram_lookup;
        Tensor inv_rms_inline27236__ssa_v1 = ext_engram_lookup;
        Tensor ret__tmp_v0_guarded_0_53 = ext_engram_lookup;
        Tensor ret__tmp_v0_guarded_1_39 = ext_engram_lookup;
        Tensor ret__tmp_v0_guarded_0_54 = ext_engram_lookup;
        Tensor ret__tmp_v0_guarded_0_55 = ext_engram_lookup;
        Tensor ret__tmp_v0_guarded_1_40 = ext_engram_lookup;
        Tensor ret__tmp_v0_guarded_0_56 = ext_engram_lookup;
        SIMPLER_SCOPE() {
            uint32_t force_topk_inline23509_ci_shapes[2] = {80, 6};
            TensorCreateInfo force_topk_inline23509_ci(force_topk_inline23509_ci_shapes, 2, DataType::INT32);
            uint32_t post_mix_inline27209_ci_shapes[2] = {80, 4};
            TensorCreateInfo post_mix_inline27209_ci(post_mix_inline27209_ci_shapes, 2, DataType::FLOAT32);
            uint32_t residual_mix_inline27210_ci_shapes[3] = {80, 4, 4};
            TensorCreateInfo residual_mix_inline27210_ci(residual_mix_inline27210_ci_shapes, 3, DataType::FLOAT32);
            TaskOutputTensors alloc_52 = alloc_tensors(force_topk_inline23509_ci, post_mix_inline27209_ci, residual_mix_inline27210_ci);
            const Tensor& force_topk_inline23509 = alloc_52.get_ref(0);
            const Tensor& post_mix_inline27209 = alloc_52.get_ref(1);
            const Tensor& residual_mix_inline27210 = alloc_52.get_ref(2);
            uint32_t layer_window_cache_inline23453_offsets[4] = {static_cast<uint32_t>((layer_inline23452 * window_blocks_per_layer_inline23440)), 0, 0, 0};
            uint32_t layer_window_cache_inline23453_shapes[4] = {(layer_window_cache_inline23453_offsets[0] >= ext_window_cache_pool.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(window_blocks_per_layer_inline23440), ext_window_cache_pool.shapes[0] - layer_window_cache_inline23453_offsets[0])), (layer_window_cache_inline23453_offsets[1] >= ext_window_cache_pool.shapes[1] ? 0u : std::min<uint32_t>(128, ext_window_cache_pool.shapes[1] - layer_window_cache_inline23453_offsets[1])), (layer_window_cache_inline23453_offsets[2] >= ext_window_cache_pool.shapes[2] ? 0u : std::min<uint32_t>(1, ext_window_cache_pool.shapes[2] - layer_window_cache_inline23453_offsets[2])), (layer_window_cache_inline23453_offsets[3] >= ext_window_cache_pool.shapes[3] ? 0u : std::min<uint32_t>(512, ext_window_cache_pool.shapes[3] - layer_window_cache_inline23453_offsets[3]))};
            Tensor layer_window_cache_inline23453 = ext_window_cache_pool.view(layer_window_cache_inline23453_shapes, layer_window_cache_inline23453_offsets);
            uint32_t layer_window_cache_scale_inline23454_offsets[4] = {static_cast<uint32_t>((layer_inline23452 * window_blocks_per_layer_inline23440)), 0, 0, 0};
            uint32_t layer_window_cache_scale_inline23454_shapes[4] = {(layer_window_cache_scale_inline23454_offsets[0] >= ext_window_cache_scale_pool.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(window_blocks_per_layer_inline23440), ext_window_cache_scale_pool.shapes[0] - layer_window_cache_scale_inline23454_offsets[0])), (layer_window_cache_scale_inline23454_offsets[1] >= ext_window_cache_scale_pool.shapes[1] ? 0u : std::min<uint32_t>(128, ext_window_cache_scale_pool.shapes[1] - layer_window_cache_scale_inline23454_offsets[1])), (layer_window_cache_scale_inline23454_offsets[2] >= ext_window_cache_scale_pool.shapes[2] ? 0u : std::min<uint32_t>(1, ext_window_cache_scale_pool.shapes[2] - layer_window_cache_scale_inline23454_offsets[2])), (layer_window_cache_scale_inline23454_offsets[3] >= ext_window_cache_scale_pool.shapes[3] ? 0u : std::min<uint32_t>(16, ext_window_cache_scale_pool.shapes[3] - layer_window_cache_scale_inline23454_offsets[3]))};
            Tensor layer_window_cache_scale_inline23454 = ext_window_cache_scale_pool.view(layer_window_cache_scale_inline23454_shapes, layer_window_cache_scale_inline23454_offsets);
            uint32_t layer_compressed_cache_inline23456_offsets[4] = {static_cast<uint32_t>((compressed_blocks_per_source_inline23441 * 3)), 0, 0, 0};
            uint32_t layer_compressed_cache_inline23456_shapes[4] = {(layer_compressed_cache_inline23456_offsets[0] >= ext_compressed_cache_pool.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(compressed_blocks_per_source_inline23441), ext_compressed_cache_pool.shapes[0] - layer_compressed_cache_inline23456_offsets[0])), (layer_compressed_cache_inline23456_offsets[1] >= ext_compressed_cache_pool.shapes[1] ? 0u : std::min<uint32_t>(128, ext_compressed_cache_pool.shapes[1] - layer_compressed_cache_inline23456_offsets[1])), (layer_compressed_cache_inline23456_offsets[2] >= ext_compressed_cache_pool.shapes[2] ? 0u : std::min<uint32_t>(1, ext_compressed_cache_pool.shapes[2] - layer_compressed_cache_inline23456_offsets[2])), (layer_compressed_cache_inline23456_offsets[3] >= ext_compressed_cache_pool.shapes[3] ? 0u : std::min<uint32_t>(256, ext_compressed_cache_pool.shapes[3] - layer_compressed_cache_inline23456_offsets[3]))};
            Tensor layer_compressed_cache_inline23456 = ext_compressed_cache_pool.view(layer_compressed_cache_inline23456_shapes, layer_compressed_cache_inline23456_offsets);
            uint32_t layer_compressed_cache_scale_inline23457_offsets[4] = {static_cast<uint32_t>((compressed_blocks_per_source_inline23441 * 3)), 0, 0, 0};
            uint32_t layer_compressed_cache_scale_inline23457_shapes[4] = {(layer_compressed_cache_scale_inline23457_offsets[0] >= ext_compressed_cache_scale_pool.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(compressed_blocks_per_source_inline23441), ext_compressed_cache_scale_pool.shapes[0] - layer_compressed_cache_scale_inline23457_offsets[0])), (layer_compressed_cache_scale_inline23457_offsets[1] >= ext_compressed_cache_scale_pool.shapes[1] ? 0u : std::min<uint32_t>(128, ext_compressed_cache_scale_pool.shapes[1] - layer_compressed_cache_scale_inline23457_offsets[1])), (layer_compressed_cache_scale_inline23457_offsets[2] >= ext_compressed_cache_scale_pool.shapes[2] ? 0u : std::min<uint32_t>(1, ext_compressed_cache_scale_pool.shapes[2] - layer_compressed_cache_scale_inline23457_offsets[2])), (layer_compressed_cache_scale_inline23457_offsets[3] >= ext_compressed_cache_scale_pool.shapes[3] ? 0u : std::min<uint32_t>(32, ext_compressed_cache_scale_pool.shapes[3] - layer_compressed_cache_scale_inline23457_offsets[3]))};
            Tensor layer_compressed_cache_scale_inline23457 = ext_compressed_cache_scale_pool.view(layer_compressed_cache_scale_inline23457_shapes, layer_compressed_cache_scale_inline23457_offsets);
            uint32_t layer_hc_fn_inline23458_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 24)), 0};
            uint32_t layer_hc_fn_inline23458_shapes[2] = {(layer_hc_fn_inline23458_offsets[0] >= ext_hc_attn_fn.shapes[0] ? 0u : std::min<uint32_t>(24, ext_hc_attn_fn.shapes[0] - layer_hc_fn_inline23458_offsets[0])), (layer_hc_fn_inline23458_offsets[1] >= ext_hc_attn_fn.shapes[1] ? 0u : std::min<uint32_t>(20480, ext_hc_attn_fn.shapes[1] - layer_hc_fn_inline23458_offsets[1]))};
            Tensor layer_hc_fn_inline23458 = ext_hc_attn_fn.view(layer_hc_fn_inline23458_shapes, layer_hc_fn_inline23458_offsets);
            uint32_t layer_hc_scale_inline23459_offsets[1] = {static_cast<uint32_t>((layer_inline23452 * 3))};
            uint32_t layer_hc_scale_inline23459_shapes[1] = {(layer_hc_scale_inline23459_offsets[0] >= ext_hc_attn_scale.shapes[0] ? 0u : std::min<uint32_t>(3, ext_hc_attn_scale.shapes[0] - layer_hc_scale_inline23459_offsets[0]))};
            Tensor layer_hc_scale_inline23459 = ext_hc_attn_scale.view(layer_hc_scale_inline23459_shapes, layer_hc_scale_inline23459_offsets);
            uint32_t layer_hc_base_inline23460_offsets[1] = {static_cast<uint32_t>((layer_inline23452 * 24))};
            uint32_t layer_hc_base_inline23460_shapes[1] = {(layer_hc_base_inline23460_offsets[0] >= ext_hc_attn_base.shapes[0] ? 0u : std::min<uint32_t>(24, ext_hc_attn_base.shapes[0] - layer_hc_base_inline23460_offsets[0]))};
            Tensor layer_hc_base_inline23460 = ext_hc_attn_base.view(layer_hc_base_inline23460_shapes, layer_hc_base_inline23460_offsets);
            uint32_t layer_norm_weight_inline23461_offsets[1] = {static_cast<uint32_t>((layer_inline23452 * 5120))};
            uint32_t layer_norm_weight_inline23461_shapes[1] = {(layer_norm_weight_inline23461_offsets[0] >= ext_attn_norm_weight.shapes[0] ? 0u : std::min<uint32_t>(5120, ext_attn_norm_weight.shapes[0] - layer_norm_weight_inline23461_offsets[0]))};
            Tensor layer_norm_weight_inline23461 = ext_attn_norm_weight.view(layer_norm_weight_inline23461_shapes, layer_norm_weight_inline23461_offsets);
            uint32_t layer_wq_a_inline23462_offsets[3] = {static_cast<uint32_t>(layer_inline23452), 0, 0};
            uint32_t layer_wq_a_inline23462_shapes[3] = {(layer_wq_a_inline23462_offsets[0] >= ext_wq_a.shapes[0] ? 0u : std::min<uint32_t>(1, ext_wq_a.shapes[0] - layer_wq_a_inline23462_offsets[0])), (layer_wq_a_inline23462_offsets[1] >= ext_wq_a.shapes[1] ? 0u : std::min<uint32_t>(5120, ext_wq_a.shapes[1] - layer_wq_a_inline23462_offsets[1])), (layer_wq_a_inline23462_offsets[2] >= ext_wq_a.shapes[2] ? 0u : std::min<uint32_t>(1280, ext_wq_a.shapes[2] - layer_wq_a_inline23462_offsets[2]))};
            Tensor layer_wq_a_inline23462_view = ext_wq_a.view(layer_wq_a_inline23462_shapes, layer_wq_a_inline23462_offsets);
            bool layer_wq_a_inline23462_empty = layer_wq_a_inline23462_view.numel() == 0;
            Tensor layer_wq_a_inline23462 = layer_wq_a_inline23462_view;
            layer_wq_a_inline23462.ndims = 2;
            layer_wq_a_inline23462.shapes[0] = layer_wq_a_inline23462_view.shapes[1];
            layer_wq_a_inline23462.strides[0] = layer_wq_a_inline23462_view.strides[1];
            layer_wq_a_inline23462.shapes[1] = layer_wq_a_inline23462_view.shapes[2];
            layer_wq_a_inline23462.strides[1] = layer_wq_a_inline23462_view.strides[2];
            if (layer_wq_a_inline23462_empty) layer_wq_a_inline23462.shapes[0] = 0;
            uint64_t layer_wq_a_inline23462_expected_stride = 1;
            bool layer_wq_a_inline23462_contiguous = true;
            for (int32_t i = static_cast<int32_t>(layer_wq_a_inline23462.ndims) - 1; i >= 0; --i) {
              layer_wq_a_inline23462_contiguous = layer_wq_a_inline23462_contiguous && (layer_wq_a_inline23462.strides[i] == layer_wq_a_inline23462_expected_stride);
              layer_wq_a_inline23462_expected_stride *= layer_wq_a_inline23462.shapes[i];
            }
            layer_wq_a_inline23462.is_contiguous = layer_wq_a_inline23462_empty || layer_wq_a_inline23462_contiguous;
            uint32_t layer_wq_a_scale_inline23463_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 160)), 0};
            uint32_t layer_wq_a_scale_inline23463_shapes[2] = {(layer_wq_a_scale_inline23463_offsets[0] >= ext_wq_a_scale.shapes[0] ? 0u : std::min<uint32_t>(160, ext_wq_a_scale.shapes[0] - layer_wq_a_scale_inline23463_offsets[0])), (layer_wq_a_scale_inline23463_offsets[1] >= ext_wq_a_scale.shapes[1] ? 0u : std::min<uint32_t>(1280, ext_wq_a_scale.shapes[1] - layer_wq_a_scale_inline23463_offsets[1]))};
            Tensor layer_wq_a_scale_inline23463 = ext_wq_a_scale.view(layer_wq_a_scale_inline23463_shapes, layer_wq_a_scale_inline23463_offsets);
            uint32_t layer_q_norm_inline23464_offsets[1] = {static_cast<uint32_t>((layer_inline23452 * 1280))};
            uint32_t layer_q_norm_inline23464_shapes[1] = {(layer_q_norm_inline23464_offsets[0] >= ext_q_norm_weight.shapes[0] ? 0u : std::min<uint32_t>(1280, ext_q_norm_weight.shapes[0] - layer_q_norm_inline23464_offsets[0]))};
            Tensor layer_q_norm_inline23464 = ext_q_norm_weight.view(layer_q_norm_inline23464_shapes, layer_q_norm_inline23464_offsets);
            uint32_t layer_wq_b_inline23465_offsets[3] = {static_cast<uint32_t>(layer_inline23452), 0, 0};
            uint32_t layer_wq_b_inline23465_shapes[3] = {(layer_wq_b_inline23465_offsets[0] >= ext_wq_b.shapes[0] ? 0u : std::min<uint32_t>(1, ext_wq_b.shapes[0] - layer_wq_b_inline23465_offsets[0])), (layer_wq_b_inline23465_offsets[1] >= ext_wq_b.shapes[1] ? 0u : std::min<uint32_t>(1280, ext_wq_b.shapes[1] - layer_wq_b_inline23465_offsets[1])), (layer_wq_b_inline23465_offsets[2] >= ext_wq_b.shapes[2] ? 0u : std::min<uint32_t>(32768, ext_wq_b.shapes[2] - layer_wq_b_inline23465_offsets[2]))};
            Tensor layer_wq_b_inline23465_view = ext_wq_b.view(layer_wq_b_inline23465_shapes, layer_wq_b_inline23465_offsets);
            bool layer_wq_b_inline23465_empty = layer_wq_b_inline23465_view.numel() == 0;
            Tensor layer_wq_b_inline23465 = layer_wq_b_inline23465_view;
            layer_wq_b_inline23465.ndims = 2;
            layer_wq_b_inline23465.shapes[0] = layer_wq_b_inline23465_view.shapes[1];
            layer_wq_b_inline23465.strides[0] = layer_wq_b_inline23465_view.strides[1];
            layer_wq_b_inline23465.shapes[1] = layer_wq_b_inline23465_view.shapes[2];
            layer_wq_b_inline23465.strides[1] = layer_wq_b_inline23465_view.strides[2];
            if (layer_wq_b_inline23465_empty) layer_wq_b_inline23465.shapes[0] = 0;
            uint64_t layer_wq_b_inline23465_expected_stride = 1;
            bool layer_wq_b_inline23465_contiguous = true;
            for (int32_t i = static_cast<int32_t>(layer_wq_b_inline23465.ndims) - 1; i >= 0; --i) {
              layer_wq_b_inline23465_contiguous = layer_wq_b_inline23465_contiguous && (layer_wq_b_inline23465.strides[i] == layer_wq_b_inline23465_expected_stride);
              layer_wq_b_inline23465_expected_stride *= layer_wq_b_inline23465.shapes[i];
            }
            layer_wq_b_inline23465.is_contiguous = layer_wq_b_inline23465_empty || layer_wq_b_inline23465_contiguous;
            uint32_t layer_wq_b_scale_inline23466_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 40)), 0};
            uint32_t layer_wq_b_scale_inline23466_shapes[2] = {(layer_wq_b_scale_inline23466_offsets[0] >= ext_wq_b_scale.shapes[0] ? 0u : std::min<uint32_t>(40, ext_wq_b_scale.shapes[0] - layer_wq_b_scale_inline23466_offsets[0])), (layer_wq_b_scale_inline23466_offsets[1] >= ext_wq_b_scale.shapes[1] ? 0u : std::min<uint32_t>(32768, ext_wq_b_scale.shapes[1] - layer_wq_b_scale_inline23466_offsets[1]))};
            Tensor layer_wq_b_scale_inline23466 = ext_wq_b_scale.view(layer_wq_b_scale_inline23466_shapes, layer_wq_b_scale_inline23466_offsets);
            uint32_t layer_wkv_inline23467_offsets[3] = {static_cast<uint32_t>(layer_inline23452), 0, 0};
            uint32_t layer_wkv_inline23467_shapes[3] = {(layer_wkv_inline23467_offsets[0] >= ext_wkv.shapes[0] ? 0u : std::min<uint32_t>(1, ext_wkv.shapes[0] - layer_wkv_inline23467_offsets[0])), (layer_wkv_inline23467_offsets[1] >= ext_wkv.shapes[1] ? 0u : std::min<uint32_t>(5120, ext_wkv.shapes[1] - layer_wkv_inline23467_offsets[1])), (layer_wkv_inline23467_offsets[2] >= ext_wkv.shapes[2] ? 0u : std::min<uint32_t>(512, ext_wkv.shapes[2] - layer_wkv_inline23467_offsets[2]))};
            Tensor layer_wkv_inline23467_view = ext_wkv.view(layer_wkv_inline23467_shapes, layer_wkv_inline23467_offsets);
            bool layer_wkv_inline23467_empty = layer_wkv_inline23467_view.numel() == 0;
            Tensor layer_wkv_inline23467 = layer_wkv_inline23467_view;
            layer_wkv_inline23467.ndims = 2;
            layer_wkv_inline23467.shapes[0] = layer_wkv_inline23467_view.shapes[1];
            layer_wkv_inline23467.strides[0] = layer_wkv_inline23467_view.strides[1];
            layer_wkv_inline23467.shapes[1] = layer_wkv_inline23467_view.shapes[2];
            layer_wkv_inline23467.strides[1] = layer_wkv_inline23467_view.strides[2];
            if (layer_wkv_inline23467_empty) layer_wkv_inline23467.shapes[0] = 0;
            uint64_t layer_wkv_inline23467_expected_stride = 1;
            bool layer_wkv_inline23467_contiguous = true;
            for (int32_t i = static_cast<int32_t>(layer_wkv_inline23467.ndims) - 1; i >= 0; --i) {
              layer_wkv_inline23467_contiguous = layer_wkv_inline23467_contiguous && (layer_wkv_inline23467.strides[i] == layer_wkv_inline23467_expected_stride);
              layer_wkv_inline23467_expected_stride *= layer_wkv_inline23467.shapes[i];
            }
            layer_wkv_inline23467.is_contiguous = layer_wkv_inline23467_empty || layer_wkv_inline23467_contiguous;
            uint32_t layer_wkv_scale_inline23468_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 160)), 0};
            uint32_t layer_wkv_scale_inline23468_shapes[2] = {(layer_wkv_scale_inline23468_offsets[0] >= ext_wkv_scale.shapes[0] ? 0u : std::min<uint32_t>(160, ext_wkv_scale.shapes[0] - layer_wkv_scale_inline23468_offsets[0])), (layer_wkv_scale_inline23468_offsets[1] >= ext_wkv_scale.shapes[1] ? 0u : std::min<uint32_t>(512, ext_wkv_scale.shapes[1] - layer_wkv_scale_inline23468_offsets[1]))};
            Tensor layer_wkv_scale_inline23468 = ext_wkv_scale.view(layer_wkv_scale_inline23468_shapes, layer_wkv_scale_inline23468_offsets);
            uint32_t layer_kv_norm_inline23469_offsets[1] = {static_cast<uint32_t>((layer_inline23452 * 512))};
            uint32_t layer_kv_norm_inline23469_shapes[1] = {(layer_kv_norm_inline23469_offsets[0] >= ext_kv_norm_weight.shapes[0] ? 0u : std::min<uint32_t>(512, ext_kv_norm_weight.shapes[0] - layer_kv_norm_inline23469_offsets[0]))};
            Tensor layer_kv_norm_inline23469 = ext_kv_norm_weight.view(layer_kv_norm_inline23469_shapes, layer_kv_norm_inline23469_offsets);
            uint32_t layer_sink_inline23470_offsets[1] = {static_cast<uint32_t>((layer_inline23452 * 64))};
            uint32_t layer_sink_inline23470_shapes[1] = {(layer_sink_inline23470_offsets[0] >= ext_attn_sink.shapes[0] ? 0u : std::min<uint32_t>(64, ext_attn_sink.shapes[0] - layer_sink_inline23470_offsets[0]))};
            Tensor layer_sink_inline23470 = ext_attn_sink.view(layer_sink_inline23470_shapes, layer_sink_inline23470_offsets);
            uint32_t layer_wo_a_inline23471_offsets[3] = {static_cast<uint32_t>((layer_inline23452 * 8)), 0, 0};
            uint32_t layer_wo_a_inline23471_shapes[3] = {(layer_wo_a_inline23471_offsets[0] >= ext_wo_a.shapes[0] ? 0u : std::min<uint32_t>(8, ext_wo_a.shapes[0] - layer_wo_a_inline23471_offsets[0])), (layer_wo_a_inline23471_offsets[1] >= ext_wo_a.shapes[1] ? 0u : std::min<uint32_t>(4096, ext_wo_a.shapes[1] - layer_wo_a_inline23471_offsets[1])), (layer_wo_a_inline23471_offsets[2] >= ext_wo_a.shapes[2] ? 0u : std::min<uint32_t>(1024, ext_wo_a.shapes[2] - layer_wo_a_inline23471_offsets[2]))};
            Tensor layer_wo_a_inline23471 = ext_wo_a.view(layer_wo_a_inline23471_shapes, layer_wo_a_inline23471_offsets);
            uint32_t layer_wo_a_scale_inline23472_offsets[3] = {static_cast<uint32_t>((layer_inline23452 * 8)), 0, 0};
            uint32_t layer_wo_a_scale_inline23472_shapes[3] = {(layer_wo_a_scale_inline23472_offsets[0] >= ext_wo_a_scale.shapes[0] ? 0u : std::min<uint32_t>(8, ext_wo_a_scale.shapes[0] - layer_wo_a_scale_inline23472_offsets[0])), (layer_wo_a_scale_inline23472_offsets[1] >= ext_wo_a_scale.shapes[1] ? 0u : std::min<uint32_t>(128, ext_wo_a_scale.shapes[1] - layer_wo_a_scale_inline23472_offsets[1])), (layer_wo_a_scale_inline23472_offsets[2] >= ext_wo_a_scale.shapes[2] ? 0u : std::min<uint32_t>(1024, ext_wo_a_scale.shapes[2] - layer_wo_a_scale_inline23472_offsets[2]))};
            Tensor layer_wo_a_scale_inline23472 = ext_wo_a_scale.view(layer_wo_a_scale_inline23472_shapes, layer_wo_a_scale_inline23472_offsets);
            uint32_t layer_wo_b_inline23473_offsets[3] = {static_cast<uint32_t>(layer_inline23452), 0, 0};
            uint32_t layer_wo_b_inline23473_shapes[3] = {(layer_wo_b_inline23473_offsets[0] >= ext_wo_b.shapes[0] ? 0u : std::min<uint32_t>(1, ext_wo_b.shapes[0] - layer_wo_b_inline23473_offsets[0])), (layer_wo_b_inline23473_offsets[1] >= ext_wo_b.shapes[1] ? 0u : std::min<uint32_t>(8192, ext_wo_b.shapes[1] - layer_wo_b_inline23473_offsets[1])), (layer_wo_b_inline23473_offsets[2] >= ext_wo_b.shapes[2] ? 0u : std::min<uint32_t>(5120, ext_wo_b.shapes[2] - layer_wo_b_inline23473_offsets[2]))};
            Tensor layer_wo_b_inline23473_view = ext_wo_b.view(layer_wo_b_inline23473_shapes, layer_wo_b_inline23473_offsets);
            bool layer_wo_b_inline23473_empty = layer_wo_b_inline23473_view.numel() == 0;
            Tensor layer_wo_b_inline23473 = layer_wo_b_inline23473_view;
            layer_wo_b_inline23473.ndims = 2;
            layer_wo_b_inline23473.shapes[0] = layer_wo_b_inline23473_view.shapes[1];
            layer_wo_b_inline23473.strides[0] = layer_wo_b_inline23473_view.strides[1];
            layer_wo_b_inline23473.shapes[1] = layer_wo_b_inline23473_view.shapes[2];
            layer_wo_b_inline23473.strides[1] = layer_wo_b_inline23473_view.strides[2];
            if (layer_wo_b_inline23473_empty) layer_wo_b_inline23473.shapes[0] = 0;
            uint64_t layer_wo_b_inline23473_expected_stride = 1;
            bool layer_wo_b_inline23473_contiguous = true;
            for (int32_t i = static_cast<int32_t>(layer_wo_b_inline23473.ndims) - 1; i >= 0; --i) {
              layer_wo_b_inline23473_contiguous = layer_wo_b_inline23473_contiguous && (layer_wo_b_inline23473.strides[i] == layer_wo_b_inline23473_expected_stride);
              layer_wo_b_inline23473_expected_stride *= layer_wo_b_inline23473.shapes[i];
            }
            layer_wo_b_inline23473.is_contiguous = layer_wo_b_inline23473_empty || layer_wo_b_inline23473_contiguous;
            uint32_t layer_wo_b_scale_inline23474_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 256)), 0};
            uint32_t layer_wo_b_scale_inline23474_shapes[2] = {(layer_wo_b_scale_inline23474_offsets[0] >= ext_wo_b_scale.shapes[0] ? 0u : std::min<uint32_t>(256, ext_wo_b_scale.shapes[0] - layer_wo_b_scale_inline23474_offsets[0])), (layer_wo_b_scale_inline23474_offsets[1] >= ext_wo_b_scale.shapes[1] ? 0u : std::min<uint32_t>(5120, ext_wo_b_scale.shapes[1] - layer_wo_b_scale_inline23474_offsets[1]))};
            Tensor layer_wo_b_scale_inline23474 = ext_wo_b_scale.view(layer_wo_b_scale_inline23474_shapes, layer_wo_b_scale_inline23474_offsets);
            if ((layer_inline23452 == 20)) {
                uint32_t post_mix_inline23519_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 4};
                TensorCreateInfo post_mix_inline23519_ci(post_mix_inline23519_ci_shapes, 2, DataType::FLOAT32);
                uint32_t residual_mix_inline23520_ci_shapes[3] = {static_cast<uint32_t>(LOCAL_T_DYN), 4, 4};
                TensorCreateInfo residual_mix_inline23520_ci(residual_mix_inline23520_ci_shapes, 3, DataType::FLOAT32);
                uint32_t x_norm_inline23524_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 5120};
                TensorCreateInfo x_norm_inline23524_ci(x_norm_inline23524_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t qr_inline23526_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 1280};
                TensorCreateInfo qr_inline23526_ci(qr_inline23526_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t query_inline23527_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 32768};
                TensorCreateInfo query_inline23527_ci(query_inline23527_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t kv_inline23528_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 512};
                TensorCreateInfo kv_inline23528_ci(kv_inline23528_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t kv_projection_inline23529_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 512};
                TensorCreateInfo kv_projection_inline23529_ci(kv_projection_inline23529_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t projected_inline23536_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 512};
                TensorCreateInfo projected_inline23536_ci(projected_inline23536_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t latent_inline23539_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 512};
                TensorCreateInfo latent_inline23539_ci(latent_inline23539_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t projected_key_inline23542_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 128};
                TensorCreateInfo projected_key_inline23542_ci(projected_key_inline23542_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t normalized_key_inline23544_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 128};
                TensorCreateInfo normalized_key_inline23544_ci(normalized_key_inline23544_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t index_key_inline23547_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 128};
                TensorCreateInfo index_key_inline23547_ci(index_key_inline23547_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t compressed_rotated_inline23550_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 512};
                TensorCreateInfo compressed_rotated_inline23550_ci(compressed_rotated_inline23550_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t projected_inline24222_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 4096};
                TensorCreateInfo projected_inline24222_ci(projected_inline24222_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t query_inline24227_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 2048};
                TensorCreateInfo query_inline24227_ci(query_inline24227_ci_shapes, 2, DataType::UINT8);
                uint32_t query_scale_inline24228_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 128};
                TensorCreateInfo query_scale_inline24228_ci(query_scale_inline24228_ci_shapes, 2, DataType::FP8E8M0);
                TaskOutputTensors alloc_53 = alloc_tensors(post_mix_inline23519_ci, residual_mix_inline23520_ci, x_norm_inline23524_ci, qr_inline23526_ci, query_inline23527_ci, kv_inline23528_ci, kv_projection_inline23529_ci, projected_inline23536_ci, latent_inline23539_ci, projected_key_inline23542_ci, normalized_key_inline23544_ci, index_key_inline23547_ci, compressed_rotated_inline23550_ci, projected_inline24222_ci, query_inline24227_ci, query_scale_inline24228_ci);
                const Tensor& post_mix_inline23519 = alloc_53.get_ref(0);
                const Tensor& residual_mix_inline23520 = alloc_53.get_ref(1);
                const Tensor& x_norm_inline23524 = alloc_53.get_ref(2);
                const Tensor& qr_inline23526 = alloc_53.get_ref(3);
                const Tensor& query_inline23527 = alloc_53.get_ref(4);
                const Tensor& kv_inline23528 = alloc_53.get_ref(5);
                const Tensor& kv_projection_inline23529 = alloc_53.get_ref(6);
                const Tensor& projected_inline23536 = alloc_53.get_ref(7);
                const Tensor& latent_inline23539 = alloc_53.get_ref(8);
                const Tensor& projected_key_inline23542 = alloc_53.get_ref(9);
                const Tensor& normalized_key_inline23544 = alloc_53.get_ref(10);
                const Tensor& index_key_inline23547 = alloc_53.get_ref(11);
                const Tensor& compressed_rotated_inline23550 = alloc_53.get_ref(12);
                const Tensor& projected_inline24222 = alloc_53.get_ref(13);
                const Tensor& query_inline24227 = alloc_53.get_ref(14);
                const Tensor& query_scale_inline24228 = alloc_53.get_ref(15);
                uint32_t weights_inline24230_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 32};
                TensorCreateInfo weights_inline24230_ci(weights_inline24230_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t partials_inline24306_ci_shapes[2] = {static_cast<uint32_t>((LOCAL_T_DYN * 4)), 32};
                TensorCreateInfo partials_inline24306_ci(partials_inline24306_ci_shapes, 2, DataType::FLOAT32);
                uint32_t runs_inline24332_ci_shapes[2] = {4, static_cast<uint32_t>((LOCAL_T_DYN + 1))};
                TensorCreateInfo runs_inline24332_ci(runs_inline24332_ci_shapes, 2, DataType::INT32);
                uint32_t attended_inline23565_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 32768};
                TensorCreateInfo attended_inline23565_ci(attended_inline23565_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t selected_inline23566_ci_shapes[3] = {static_cast<uint32_t>(LOCAL_T_DYN), 640, 512};
                TensorCreateInfo selected_inline23566_ci(selected_inline23566_ci_shapes, 3, DataType::BFLOAT16);
                uint32_t indices_inline23567_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 640};
                TensorCreateInfo indices_inline23567_ci(indices_inline23567_ci_shapes, 2, DataType::INT32);
                uint32_t attn_out_inline23572_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 5120};
                TensorCreateInfo attn_out_inline23572_ci(attn_out_inline23572_ci_shapes, 2, DataType::BFLOAT16);
                TaskOutputTensors alloc_54 = alloc_tensors(weights_inline24230_ci, partials_inline24306_ci, runs_inline24332_ci, attended_inline23565_ci, selected_inline23566_ci, indices_inline23567_ci, attn_out_inline23572_ci);
                const Tensor& weights_inline24230 = alloc_54.get_ref(0);
                const Tensor& partials_inline24306 = alloc_54.get_ref(1);
                const Tensor& runs_inline24332 = alloc_54.get_ref(2);
                const Tensor& attended_inline23565 = alloc_54.get_ref(3);
                const Tensor& selected_inline23566 = alloc_54.get_ref(4);
                const Tensor& indices_inline23567 = alloc_54.get_ref(5);
                const Tensor& attn_out_inline23572 = alloc_54.get_ref(6);
                uint32_t full_index_cache_inline23476_offsets[4] = {static_cast<uint32_t>((index_blocks_per_source_inline23442 * 3)), 0, 0, 0};
                uint32_t full_index_cache_inline23476_shapes[4] = {(full_index_cache_inline23476_offsets[0] >= ext_index_cache_pool.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(index_blocks_per_source_inline23442), ext_index_cache_pool.shapes[0] - full_index_cache_inline23476_offsets[0])), (full_index_cache_inline23476_offsets[1] >= ext_index_cache_pool.shapes[1] ? 0u : std::min<uint32_t>(128, ext_index_cache_pool.shapes[1] - full_index_cache_inline23476_offsets[1])), (full_index_cache_inline23476_offsets[2] >= ext_index_cache_pool.shapes[2] ? 0u : std::min<uint32_t>(1, ext_index_cache_pool.shapes[2] - full_index_cache_inline23476_offsets[2])), (full_index_cache_inline23476_offsets[3] >= ext_index_cache_pool.shapes[3] ? 0u : std::min<uint32_t>(64, ext_index_cache_pool.shapes[3] - full_index_cache_inline23476_offsets[3]))};
                Tensor full_index_cache_inline23476 = ext_index_cache_pool.view(full_index_cache_inline23476_shapes, full_index_cache_inline23476_offsets);
                uint32_t full_index_cache_scale_inline23477_offsets[4] = {static_cast<uint32_t>((index_blocks_per_source_inline23442 * 3)), 0, 0, 0};
                uint32_t full_index_cache_scale_inline23477_shapes[4] = {(full_index_cache_scale_inline23477_offsets[0] >= ext_index_cache_scale_pool.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(index_blocks_per_source_inline23442), ext_index_cache_scale_pool.shapes[0] - full_index_cache_scale_inline23477_offsets[0])), (full_index_cache_scale_inline23477_offsets[1] >= ext_index_cache_scale_pool.shapes[1] ? 0u : std::min<uint32_t>(128, ext_index_cache_scale_pool.shapes[1] - full_index_cache_scale_inline23477_offsets[1])), (full_index_cache_scale_inline23477_offsets[2] >= ext_index_cache_scale_pool.shapes[2] ? 0u : std::min<uint32_t>(1, ext_index_cache_scale_pool.shapes[2] - full_index_cache_scale_inline23477_offsets[2])), (full_index_cache_scale_inline23477_offsets[3] >= ext_index_cache_scale_pool.shapes[3] ? 0u : std::min<uint32_t>(4, ext_index_cache_scale_pool.shapes[3] - full_index_cache_scale_inline23477_offsets[3]))};
                Tensor full_index_cache_scale_inline23477 = ext_index_cache_scale_pool.view(full_index_cache_scale_inline23477_shapes, full_index_cache_scale_inline23477_offsets);
                uint32_t full_index_wq_b_scale_inline23478_offsets[2] = {120, 0};
                uint32_t full_index_wq_b_scale_inline23478_shapes[2] = {(full_index_wq_b_scale_inline23478_offsets[0] >= ext_index_wq_b_scale.shapes[0] ? 0u : std::min<uint32_t>(40, ext_index_wq_b_scale.shapes[0] - full_index_wq_b_scale_inline23478_offsets[0])), (full_index_wq_b_scale_inline23478_offsets[1] >= ext_index_wq_b_scale.shapes[1] ? 0u : std::min<uint32_t>(4096, ext_index_wq_b_scale.shapes[1] - full_index_wq_b_scale_inline23478_offsets[1]))};
                Tensor full_index_wq_b_scale_inline23478 = ext_index_wq_b_scale.view(full_index_wq_b_scale_inline23478_shapes, full_index_wq_b_scale_inline23478_offsets);
                uint32_t full_compressor_norm_inline23479_offsets[1] = {1536};
                uint32_t full_compressor_norm_inline23479_shapes[1] = {(full_compressor_norm_inline23479_offsets[0] >= ext_compressor_norm_weight.shapes[0] ? 0u : std::min<uint32_t>(512, ext_compressor_norm_weight.shapes[0] - full_compressor_norm_inline23479_offsets[0]))};
                Tensor full_compressor_norm_inline23479 = ext_compressor_norm_weight.view(full_compressor_norm_inline23479_shapes, full_compressor_norm_inline23479_offsets);
                uint32_t full_index_wk_inline23480_offsets[5] = {3, 0, 0, 0, 0};
                uint32_t full_index_wk_inline23480_shapes[5] = {(full_index_wk_inline23480_offsets[0] >= ext_index_wk.shapes[0] ? 0u : std::min<uint32_t>(1, ext_index_wk.shapes[0] - full_index_wk_inline23480_offsets[0])), (full_index_wk_inline23480_offsets[1] >= ext_index_wk.shapes[1] ? 0u : std::min<uint32_t>(8, ext_index_wk.shapes[1] - full_index_wk_inline23480_offsets[1])), (full_index_wk_inline23480_offsets[2] >= ext_index_wk.shapes[2] ? 0u : std::min<uint32_t>(32, ext_index_wk.shapes[2] - full_index_wk_inline23480_offsets[2])), (full_index_wk_inline23480_offsets[3] >= ext_index_wk.shapes[3] ? 0u : std::min<uint32_t>(16, ext_index_wk.shapes[3] - full_index_wk_inline23480_offsets[3])), (full_index_wk_inline23480_offsets[4] >= ext_index_wk.shapes[4] ? 0u : std::min<uint32_t>(16, ext_index_wk.shapes[4] - full_index_wk_inline23480_offsets[4]))};
                Tensor full_index_wk_inline23480 = ext_index_wk.view(full_index_wk_inline23480_shapes, full_index_wk_inline23480_offsets);
                uint32_t full_index_norm_inline23481_offsets[1] = {384};
                uint32_t full_index_norm_inline23481_shapes[1] = {(full_index_norm_inline23481_offsets[0] >= ext_index_norm_weight.shapes[0] ? 0u : std::min<uint32_t>(128, ext_index_norm_weight.shapes[0] - full_index_norm_inline23481_offsets[0]))};
                Tensor full_index_norm_inline23481 = ext_index_norm_weight.view(full_index_norm_inline23481_shapes, full_index_norm_inline23481_offsets);
                uint32_t full_index_wq_b_inline23482_offsets[3] = {3, 0, 0};
                uint32_t full_index_wq_b_inline23482_shapes[3] = {(full_index_wq_b_inline23482_offsets[0] >= ext_index_wq_b.shapes[0] ? 0u : std::min<uint32_t>(1, ext_index_wq_b.shapes[0] - full_index_wq_b_inline23482_offsets[0])), (full_index_wq_b_inline23482_offsets[1] >= ext_index_wq_b.shapes[1] ? 0u : std::min<uint32_t>(1280, ext_index_wq_b.shapes[1] - full_index_wq_b_inline23482_offsets[1])), (full_index_wq_b_inline23482_offsets[2] >= ext_index_wq_b.shapes[2] ? 0u : std::min<uint32_t>(4096, ext_index_wq_b.shapes[2] - full_index_wq_b_inline23482_offsets[2]))};
                Tensor full_index_wq_b_inline23482_view = ext_index_wq_b.view(full_index_wq_b_inline23482_shapes, full_index_wq_b_inline23482_offsets);
                bool full_index_wq_b_inline23482_empty = full_index_wq_b_inline23482_view.numel() == 0;
                Tensor full_index_wq_b_inline23482 = full_index_wq_b_inline23482_view;
                full_index_wq_b_inline23482.ndims = 2;
                full_index_wq_b_inline23482.shapes[0] = full_index_wq_b_inline23482_view.shapes[1];
                full_index_wq_b_inline23482.strides[0] = full_index_wq_b_inline23482_view.strides[1];
                full_index_wq_b_inline23482.shapes[1] = full_index_wq_b_inline23482_view.shapes[2];
                full_index_wq_b_inline23482.strides[1] = full_index_wq_b_inline23482_view.strides[2];
                if (full_index_wq_b_inline23482_empty) full_index_wq_b_inline23482.shapes[0] = 0;
                uint64_t full_index_wq_b_inline23482_expected_stride = 1;
                bool full_index_wq_b_inline23482_contiguous = true;
                for (int32_t i = static_cast<int32_t>(full_index_wq_b_inline23482.ndims) - 1; i >= 0; --i) {
                  full_index_wq_b_inline23482_contiguous = full_index_wq_b_inline23482_contiguous && (full_index_wq_b_inline23482.strides[i] == full_index_wq_b_inline23482_expected_stride);
                  full_index_wq_b_inline23482_expected_stride *= full_index_wq_b_inline23482.shapes[i];
                }
                full_index_wq_b_inline23482.is_contiguous = full_index_wq_b_inline23482_empty || full_index_wq_b_inline23482_contiguous;
                uint32_t full_index_weights_proj_inline23483_offsets[5] = {3, 0, 0, 0, 0};
                uint32_t full_index_weights_proj_inline23483_shapes[5] = {(full_index_weights_proj_inline23483_offsets[0] >= ext_index_weights_proj.shapes[0] ? 0u : std::min<uint32_t>(1, ext_index_weights_proj.shapes[0] - full_index_weights_proj_inline23483_offsets[0])), (full_index_weights_proj_inline23483_offsets[1] >= ext_index_weights_proj.shapes[1] ? 0u : std::min<uint32_t>(2, ext_index_weights_proj.shapes[1] - full_index_weights_proj_inline23483_offsets[1])), (full_index_weights_proj_inline23483_offsets[2] >= ext_index_weights_proj.shapes[2] ? 0u : std::min<uint32_t>(320, ext_index_weights_proj.shapes[2] - full_index_weights_proj_inline23483_offsets[2])), (full_index_weights_proj_inline23483_offsets[3] >= ext_index_weights_proj.shapes[3] ? 0u : std::min<uint32_t>(16, ext_index_weights_proj.shapes[3] - full_index_weights_proj_inline23483_offsets[3])), (full_index_weights_proj_inline23483_offsets[4] >= ext_index_weights_proj.shapes[4] ? 0u : std::min<uint32_t>(16, ext_index_weights_proj.shapes[4] - full_index_weights_proj_inline23483_offsets[4]))};
                Tensor full_index_weights_proj_inline23483 = ext_index_weights_proj.view(full_index_weights_proj_inline23483_shapes, full_index_weights_proj_inline23483_offsets);
                int64_t t_linear_inline23575 = (((LOCAL_T_DYN + 15) / 16) * 16);
                uint32_t x_flat_inline23576_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                Tensor x_flat_inline23576 = ext_x_hc.reshape(x_flat_inline23576_shapes, 2);
                uint32_t inv_rms_inline23577_ci_shapes[2] = {static_cast<uint32_t>(t_linear_inline23575), 1};
                TensorCreateInfo inv_rms_inline23577_ci(inv_rms_inline23577_ci_shapes, 2, DataType::FLOAT32);
                TaskOutputTensors alloc_55 = alloc_tensors(inv_rms_inline23577_ci);
                const Tensor& inv_rms_inline23577 = alloc_55.get_ref(0);
                int64_t spmd_core_num_13 = ((LOCAL_T_DYN + 7) / 8);
                Tensor inv_rms_inline23577__ssa_v1 = ext_engram_lookup;
                if ((0 < spmd_core_num_13)) {

                    // Spmd mhc_rms_spmd_3: mhc_rms
                    CoreTaskArgs params_t132;
                    params_t132.add_input(x_flat_inline23576);
                    params_t132.add_inout(inv_rms_inline23577);
                    params_t132.add_scalar(LOCAL_T_DYN);
                    params_t132.launch_spec.set_block_num(spmd_core_num_13);
                    rt_submit_aiv_task(28, params_t132);
                    inv_rms_inline23577__ssa_v1 = inv_rms_inline23577;
                } else {
                    inv_rms_inline23577__ssa_v1 = inv_rms_inline23577;
                }
                uint32_t mixes_partials_inline23586_ci_shapes[2] = {static_cast<uint32_t>((t_linear_inline23575 * 4)), 32};
                TensorCreateInfo mixes_partials_inline23586_ci(mixes_partials_inline23586_ci_shapes, 2, DataType::FLOAT32);
                TaskOutputTensors alloc_56 = alloc_tensors(mixes_partials_inline23586_ci);
                const Tensor& mixes_partials_inline23586 = alloc_56.get_ref(0);
                int64_t tasks_inline23587 = (((LOCAL_T_DYN + 79) / 80) * 4);
                int32_t t__tmp_v760 = rt_available_cluster_count();
                int64_t workers_inline23588 = std::min<int64_t>(tasks_inline23587, static_cast<int64_t>(t__tmp_v760));
                if ((0 < workers_inline23588)) {

                    // Spmd mhc_linear_3: _hc_linear
                    CoreTaskArgs params_t133;
                    params_t133.add_input(x_flat_inline23576);
                    params_t133.add_input(layer_hc_fn_inline23458);
                    params_t133.add_inout(mixes_partials_inline23586);
                    params_t133.add_scalar(LOCAL_T_DYN);
                    params_t133.add_scalar(t_linear_inline23575);
                    params_t133.launch_spec.set_block_num(workers_inline23588);
                    rt_submit_aic_task(29, params_t133);
                } else {
                }
                Tensor inv_rms_inline23522 = inv_rms_inline23577__ssa_v1;
                Tensor partials_inline23523 = mixes_partials_inline23586;
                int64_t linear_tokens_inline23598 = (int64_t)inv_rms_inline23522.shapes[0];
                uint32_t base_row_inline23599_shapes[2] = {1, 24};
                Tensor base_row_inline23599 = layer_hc_base_inline23460.reshape(base_row_inline23599_shapes, 2);
                uint32_t residual_flat_inline23600_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 16};
                Tensor residual_flat_inline23600 = residual_mix_inline23520.reshape(residual_flat_inline23600_shapes, 2);
                uint32_t indices_scale0_inline23601[1] = {0};
                float scale0_inline23601 = get_tensor_data<float>(layer_hc_scale_inline23459, 1, indices_scale0_inline23601);
                uint32_t indices_scale1_inline23602[1] = {1};
                float scale1_inline23602 = get_tensor_data<float>(layer_hc_scale_inline23459, 1, indices_scale1_inline23602);
                uint32_t indices_scale2_inline23603[1] = {2};
                float scale2_inline23603 = get_tensor_data<float>(layer_hc_scale_inline23459, 1, indices_scale2_inline23603);
                int64_t blocks_inline23604 = ((LOCAL_T_DYN + 1) / 2);
                int64_t workers_inline23605 = std::min<int64_t>(blocks_inline23604, 12);
                Tensor ret__tmp_v0_guarded_0_27 = ext_engram_lookup;
                Tensor ret__tmp_v0_guarded_1_19 = ext_engram_lookup;
                if ((0 < workers_inline23605)) {

                    // Spmd mhc_sinkhorn_spmd_3: mhc_sinkhorn
                    CoreTaskArgs params_t134;
                    params_t134.add_input(partials_inline23523);
                    params_t134.add_input(inv_rms_inline23522);
                    params_t134.add_input(base_row_inline23599);
                    params_t134.add_inout(attention_pre_mix_inline23444__rv_v3);
                    params_t134.add_inout(post_mix_inline23519);
                    params_t134.add_inout(residual_flat_inline23600);
                    params_t134.add_scalar(LOCAL_T_DYN);
                    params_t134.add_scalar(blocks_inline23604);
                    params_t134.add_scalar(workers_inline23605);
                    params_t134.add_scalar(linear_tokens_inline23598);
                    params_t134.add_scalar(to_u64(scale0_inline23601));
                    params_t134.add_scalar(to_u64(scale1_inline23602));
                    params_t134.add_scalar(to_u64(scale2_inline23603));
                    params_t134.launch_spec.set_block_num(workers_inline23605);
                    TaskOutputTensors task_134_outs = rt_submit_aiv_task(30, params_t134);
                    ret__tmp_v0_guarded_0_27 = attention_pre_mix_inline23444__rv_v3;
                    ret__tmp_v0_guarded_1_19 = post_mix_inline23519;
                } else {

                    // Phase-fence barrier 26: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_26;
                    TaskOutputTensors phase_fence_barrier_26_outs = rt_submit_dummy_task(params_phase_fence_barrier_26);
                    TaskId spmd_empty_tid_21 = phase_fence_barrier_26_outs.task_id();
                    ret__tmp_v0_guarded_0_27 = attention_pre_mix_inline23444__rv_v3;
                    ret__tmp_v0_guarded_1_19 = post_mix_inline23519;
                }
                Tensor attention_pre_mix_inline23444__ssa_v1 = ret__tmp_v0_guarded_0_27;
                Tensor post_mix_inline23519__ssa_v1 = ret__tmp_v0_guarded_1_19;
                uint32_t x_flat_inline23710_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                Tensor x_flat_inline23710 = ext_x_hc.reshape(x_flat_inline23710_shapes, 2);
                int64_t spmd_core_num_14 = ((LOCAL_T_DYN + 7) / 8);
                Tensor ret__tmp_v0_guarded_0_28 = ext_engram_lookup;
                TaskId ret__tmp_v0_guarded_1_20 = TaskId::invalid();
                if ((0 < spmd_core_num_14)) {

                    // Spmd hc_collapse_norm_spmd_2: hc_collapse_norm
                    CoreTaskArgs params_t135;
                    params_t135.add_input(encoder_pre_mix__rv_v1);
                    params_t135.add_input(x_flat_inline23710);
                    params_t135.add_inout(x_norm_inline23524);
                    params_t135.add_input(layer_norm_weight_inline23461);
                    params_t135.add_scalar(LOCAL_T_DYN);
                    params_t135.launch_spec.set_block_num(spmd_core_num_14);
                    TaskOutputTensors task_135_outs = rt_submit_aiv_task(8, params_t135);
                    TaskId ret__tmp_v0_nonempty_1_22 = task_135_outs.task_id();
                    ret__tmp_v0_guarded_0_28 = x_norm_inline23524;
                    ret__tmp_v0_guarded_1_20 = ret__tmp_v0_nonempty_1_22;
                } else {

                    // Phase-fence barrier 27: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_27;
                    TaskOutputTensors phase_fence_barrier_27_outs = rt_submit_dummy_task(params_phase_fence_barrier_27);
                    TaskId spmd_empty_tid_22 = phase_fence_barrier_27_outs.task_id();
                    ret__tmp_v0_guarded_0_28 = x_norm_inline23524;
                    ret__tmp_v0_guarded_1_20 = spmd_empty_tid_22;
                }
                Tensor x_norm_inline23524__rv_v2 = ret__tmp_v0_guarded_0_28;
                TaskId ready_inline23711 = ret__tmp_v0_guarded_1_20;
                TaskId x_ready_inline23525 = ready_inline23711;
                int64_t padded_rows_inline23758 = (((LOCAL_T_DYN + 79) / 80) * 80);
                uint32_t payload_inline23759_ci_shapes[2] = {static_cast<uint32_t>(padded_rows_inline23758), 5120};
                TensorCreateInfo payload_inline23759_ci(payload_inline23759_ci_shapes, 2, DataType::FP8E4M3FN);
                TaskOutputTensors alloc_57 = alloc_tensors(payload_inline23759_ci);
                const Tensor& payload_inline23759 = alloc_57.get_ref(0);
                uint32_t scales_inline23760_ci_shapes[2] = {1, static_cast<uint32_t>((padded_rows_inline23758 * 160))};
                TensorCreateInfo scales_inline23760_ci(scales_inline23760_ci_shapes, 2, DataType::FP8E8M0);
                TaskOutputTensors alloc_58 = alloc_tensors(scales_inline23760_ci);
                const Tensor& scales_inline23760 = alloc_58.get_ref(0);
                int64_t quant_blocks_inline23761 = ((padded_rows_inline23758 / 16) * 20);
                int32_t t__tmp_v770 = rt_available_cluster_count();
                int64_t workers_inline23762 = std::min<int64_t>(std::min<int64_t>(quant_blocks_inline23761, (static_cast<int64_t>(t__tmp_v770) * 2)), 32);
                Tensor ret__tmp_v0_guarded_0_29 = ext_engram_lookup;
                Tensor ret__tmp_v0_guarded_1_21 = ext_engram_lookup;
                if ((0 < workers_inline23762)) {

                    // Spmd attention_qkv_input_quant_pooled_spmd_2: attention_qkv_input_quant_pooled
                    CoreTaskArgs params_t136;
                    params_t136.add_inout(payload_inline23759);
                    params_t136.add_inout(scales_inline23760);
                    params_t136.add_input(x_norm_inline23524__rv_v2);
                    params_t136.add_scalar(quant_blocks_inline23761);
                    params_t136.add_scalar(workers_inline23762);
                    params_t136.add_scalar(attention_count_inline23434);
                    params_t136.launch_spec.set_block_num(workers_inline23762);
                    rt_submit_aiv_task(10, params_t136);
                    ret__tmp_v0_guarded_0_29 = payload_inline23759;
                    ret__tmp_v0_guarded_1_21 = scales_inline23760;
                } else {
                    ret__tmp_v0_guarded_0_29 = payload_inline23759;
                    ret__tmp_v0_guarded_1_21 = scales_inline23760;
                }
                Tensor payload_inline23759__rv_v2 = ret__tmp_v0_guarded_0_29;
                Tensor scales_inline23760__rv_v2 = ret__tmp_v0_guarded_1_21;
                Tensor proj_payload_inline23531 = payload_inline23759__rv_v2;
                Tensor proj_scales_inline23532 = scales_inline23760__rv_v2;
                Tensor query_inline23805 = query_inline23527;
                int64_t payload_rows_inline23806 = (int64_t)proj_payload_inline23531.shapes[0];
                uint32_t qb_payload_inline23807_ci_shapes[2] = {static_cast<uint32_t>(payload_rows_inline23806), 1280};
                TensorCreateInfo qb_payload_inline23807_ci(qb_payload_inline23807_ci_shapes, 2, DataType::FP8E4M3FN);
                TaskOutputTensors alloc_59 = alloc_tensors(qb_payload_inline23807_ci);
                const Tensor& qb_payload_inline23807 = alloc_59.get_ref(0);
                uint32_t qb_scales_inline23808_ci_shapes[2] = {1, static_cast<uint32_t>((payload_rows_inline23806 * 40))};
                TensorCreateInfo qb_scales_inline23808_ci(qb_scales_inline23808_ci_shapes, 2, DataType::FP8E8M0);
                TaskOutputTensors alloc_60 = alloc_tensors(qb_scales_inline23808_ci);
                const Tensor& qb_scales_inline23808 = alloc_60.get_ref(0);
                Tensor query_inline23805__rv_v2 = query_inline23805;
                for (int64_t group_start_inline23809 = 0; group_start_inline23809 < attention_count_inline23434; group_start_inline23809 += 80) {
                    uint32_t projected_inline23814_ci_shapes[2] = {80, 1280};
                    TensorCreateInfo projected_inline23814_ci(projected_inline23814_ci_shapes, 2, DataType::BFLOAT16);
                    TaskOutputTensors alloc_61 = alloc_tensors(projected_inline23814_ci);
                    const Tensor& projected_inline23814 = alloc_61.get_ref(0);
                    int64_t group_rows_inline23810 = std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) - group_start_inline23809), 80);
                    uint32_t group_payload_inline23811_offsets[2] = {static_cast<uint32_t>(group_start_inline23809), 0};
                    uint32_t group_payload_inline23811_shapes[2] = {(group_payload_inline23811_offsets[0] >= proj_payload_inline23531.shapes[0] ? 0u : std::min<uint32_t>(80, proj_payload_inline23531.shapes[0] - group_payload_inline23811_offsets[0])), (group_payload_inline23811_offsets[1] >= proj_payload_inline23531.shapes[1] ? 0u : std::min<uint32_t>(5120, proj_payload_inline23531.shapes[1] - group_payload_inline23811_offsets[1]))};
                    Tensor group_payload_inline23811 = proj_payload_inline23531.view(group_payload_inline23811_shapes, group_payload_inline23811_offsets);
                    int64_t scale_start_inline23812 = (group_start_inline23809 * 160);
                    uint32_t group_scales_inline23813_offsets[2] = {0, static_cast<uint32_t>(scale_start_inline23812)};
                    uint32_t group_scales_inline23813_shapes[2] = {(group_scales_inline23813_offsets[0] >= proj_scales_inline23532.shapes[0] ? 0u : std::min<uint32_t>(1, proj_scales_inline23532.shapes[0] - group_scales_inline23813_offsets[0])), (group_scales_inline23813_offsets[1] >= proj_scales_inline23532.shapes[1] ? 0u : std::min<uint32_t>(12800, proj_scales_inline23532.shapes[1] - group_scales_inline23813_offsets[1]))};
                    Tensor group_scales_inline23813 = proj_scales_inline23532.view(group_scales_inline23813_shapes, group_scales_inline23813_offsets);

                    // Spmd proj_q_a_2: _proj_q_a
                    CoreTaskArgs params_t137;
                    params_t137.add_input(group_payload_inline23811);
                    params_t137.add_input(layer_wq_a_inline23462);
                    params_t137.add_input(group_scales_inline23813);
                    params_t137.add_input(layer_wq_a_scale_inline23463);
                    params_t137.add_inout(projected_inline23814);
                    params_t137.add_scalar(group_rows_inline23810);
                    params_t137.launch_spec.set_block_num(20);
                    rt_submit_aic_task(11, params_t137);
                    uint32_t qb_payload_group_inline23815_offsets[2] = {static_cast<uint32_t>(group_start_inline23809), 0};
                    uint32_t qb_payload_group_inline23815_shapes[2] = {(qb_payload_group_inline23815_offsets[0] >= qb_payload_inline23807.shapes[0] ? 0u : std::min<uint32_t>(80, qb_payload_inline23807.shapes[0] - qb_payload_group_inline23815_offsets[0])), (qb_payload_group_inline23815_offsets[1] >= qb_payload_inline23807.shapes[1] ? 0u : std::min<uint32_t>(1280, qb_payload_inline23807.shapes[1] - qb_payload_group_inline23815_offsets[1]))};
                    Tensor qb_payload_group_inline23815 = qb_payload_inline23807.view(qb_payload_group_inline23815_shapes, qb_payload_group_inline23815_offsets);
                    int64_t qb_scale_start_inline23816 = (group_start_inline23809 * 40);
                    uint32_t qb_scales_group_inline23817_offsets[2] = {0, static_cast<uint32_t>(qb_scale_start_inline23816)};
                    uint32_t qb_scales_group_inline23817_shapes[2] = {(qb_scales_group_inline23817_offsets[0] >= qb_scales_inline23808.shapes[0] ? 0u : std::min<uint32_t>(1, qb_scales_inline23808.shapes[0] - qb_scales_group_inline23817_offsets[0])), (qb_scales_group_inline23817_offsets[1] >= qb_scales_inline23808.shapes[1] ? 0u : std::min<uint32_t>(3200, qb_scales_inline23808.shapes[1] - qb_scales_group_inline23817_offsets[1]))};
                    Tensor qb_scales_group_inline23817 = qb_scales_inline23808.view(qb_scales_group_inline23817_shapes, qb_scales_group_inline23817_offsets);
                    for (int64_t q_mt_inline23818 = 0; q_mt_inline23818 < 5; q_mt_inline23818 += 1) {
                        int64_t q_local_inline23819 = (q_mt_inline23818 * 16);
                        int64_t q_start_inline23820 = (group_start_inline23809 + std::min<int64_t>(q_local_inline23819, group_rows_inline23810));
                        int64_t q_rows_inline23821 = std::max<int64_t>(std::min<int64_t>((group_rows_inline23810 - q_local_inline23819), 16), 0);
                        uint32_t projected_tile_inline23822_offsets[2] = {static_cast<uint32_t>(q_local_inline23819), 0};
                        uint32_t projected_tile_inline23822_shapes[2] = {(projected_tile_inline23822_offsets[0] >= projected_inline23814.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(q_rows_inline23821), projected_inline23814.shapes[0] - projected_tile_inline23822_offsets[0])), (projected_tile_inline23822_offsets[1] >= projected_inline23814.shapes[1] ? 0u : std::min<uint32_t>(1280, projected_inline23814.shapes[1] - projected_tile_inline23822_offsets[1]))};
                        Tensor projected_tile_inline23822 = projected_inline23814.view(projected_tile_inline23822_shapes, projected_tile_inline23822_offsets);
                        uint32_t latent_tile_inline23823_offsets[2] = {static_cast<uint32_t>(q_start_inline23820), 0};
                        uint32_t latent_tile_inline23823_shapes[2] = {(latent_tile_inline23823_offsets[0] >= qr_inline23526.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(q_rows_inline23821), qr_inline23526.shapes[0] - latent_tile_inline23823_offsets[0])), (latent_tile_inline23823_offsets[1] >= qr_inline23526.shapes[1] ? 0u : std::min<uint32_t>(1280, qr_inline23526.shapes[1] - latent_tile_inline23823_offsets[1]))};
                        Tensor latent_tile_inline23823 = qr_inline23526.view(latent_tile_inline23823_shapes, latent_tile_inline23823_offsets);
                        uint32_t qb_payload_tile_inline23824_offsets[2] = {static_cast<uint32_t>(q_local_inline23819), 0};
                        uint32_t qb_payload_tile_inline23824_shapes[2] = {(qb_payload_tile_inline23824_offsets[0] >= qb_payload_group_inline23815.shapes[0] ? 0u : std::min<uint32_t>(16, qb_payload_group_inline23815.shapes[0] - qb_payload_tile_inline23824_offsets[0])), (qb_payload_tile_inline23824_offsets[1] >= qb_payload_group_inline23815.shapes[1] ? 0u : std::min<uint32_t>(1280, qb_payload_group_inline23815.shapes[1] - qb_payload_tile_inline23824_offsets[1]))};
                        Tensor qb_payload_tile_inline23824 = qb_payload_group_inline23815.view(qb_payload_tile_inline23824_shapes, qb_payload_tile_inline23824_offsets);
                        int64_t qb_scale_offset_inline23825 = (q_local_inline23819 * 40);
                        uint32_t qb_scales_tile_inline23826_offsets[2] = {0, static_cast<uint32_t>(qb_scale_offset_inline23825)};
                        uint32_t qb_scales_tile_inline23826_shapes[2] = {(qb_scales_tile_inline23826_offsets[0] >= qb_scales_group_inline23817.shapes[0] ? 0u : std::min<uint32_t>(1, qb_scales_group_inline23817.shapes[0] - qb_scales_tile_inline23826_offsets[0])), (qb_scales_tile_inline23826_offsets[1] >= qb_scales_group_inline23817.shapes[1] ? 0u : std::min<uint32_t>(640, qb_scales_group_inline23817.shapes[1] - qb_scales_tile_inline23826_offsets[1]))};
                        Tensor qb_scales_tile_inline23826 = qb_scales_group_inline23817.view(qb_scales_tile_inline23826_shapes, qb_scales_tile_inline23826_offsets);
                        uint32_t weight_row_inline23831_shapes[2] = {1, 1280};
                        Tensor weight_row_inline23831 = layer_q_norm_inline23464.reshape(weight_row_inline23831_shapes, 2);

                        // Spmd attention_q_norm_quant_spmd_2: attention_q_norm_quant_2
                        CoreTaskArgs params_t138;
                        params_t138.add_input(projected_tile_inline23822);
                        params_t138.add_input(weight_row_inline23831);
                        params_t138.add_inout(latent_tile_inline23823);
                        params_t138.add_inout(qb_payload_tile_inline23824);
                        params_t138.add_inout(qb_scales_tile_inline23826);
                        params_t138.add_scalar(q_rows_inline23821);
                        params_t138.launch_spec.set_block_num(5);
                        rt_submit_aiv_task(94, params_t138);
                    }
                    int32_t group_tokens_inline23827 = static_cast<int32_t>(group_rows_inline23810);
                    int32_t t__tmp_v779 = rt_available_cluster_count();
                    int64_t qb_workers_v1_inline23829 = std::min<int64_t>(static_cast<int64_t>(t__tmp_v779), 32);
                    int64_t qb_workers_v2_inline23830 = std::max<int64_t>(qb_workers_v1_inline23829, 1);

                    // Spmd proj_q_b_2: _proj_q_b
                    CoreTaskArgs params_t139;
                    params_t139.add_input(qb_payload_group_inline23815);
                    params_t139.add_input(layer_wq_b_inline23465);
                    params_t139.add_input(qb_scales_group_inline23817);
                    params_t139.add_input(layer_wq_b_scale_inline23466);
                    params_t139.add_inout(query_inline23805__rv_v2);
                    params_t139.add_scalar(group_rows_inline23810);
                    params_t139.add_scalar((uint64_t)32768);
                    params_t139.add_scalar(group_start_inline23809);
                    params_t139.launch_spec.set_block_num(qb_workers_v2_inline23830);
                    rt_submit_aic_task(13, params_t139);
                    int32_t t__tmp_v780 = rt_available_cluster_count();
                    int64_t rope_workers_inline23891 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v780) * 2), 32);
                    Tensor query_inline23805__rv_v5 = ext_engram_lookup;
                    if ((0 < rope_workers_inline23891)) {

                        // Spmd attention_q_rope_spmd_2: attention_q_rope
                        CoreTaskArgs params_t140;
                        params_t140.add_inout(query_inline23805__rv_v2);
                        params_t140.add_input(compressed_token_rope_cos_inline23437);
                        params_t140.add_input(compressed_token_rope_sin_inline23438);
                        params_t140.add_scalar(rope_workers_inline23891);
                        params_t140.add_scalar(group_start_inline23809);
                        params_t140.add_scalar(group_tokens_inline23827);
                        params_t140.launch_spec.set_block_num(rope_workers_inline23891);
                        rt_submit_aiv_task(14, params_t140);
                        query_inline23805__rv_v5 = query_inline23805__rv_v2;
                    } else {
                        query_inline23805__rv_v5 = query_inline23805__rv_v2;
                    }
                    query_inline23805__rv_v2 = query_inline23805__rv_v5;
                }
                Tensor query_inline23527__ssa_v1 = query_inline23805__rv_v2;
                Tensor qb_payload_inline23534 = qb_payload_inline23807;
                Tensor qb_scales_inline23535 = qb_scales_inline23808;
                Tensor kv_projection_inline23925 = kv_projection_inline23529;
                Tensor kv_inline23528__rv_v2 = kv_inline23528;
                for (int64_t group_start_inline23926 = 0; group_start_inline23926 < attention_count_inline23434; group_start_inline23926 += 80) {
                    int64_t group_rows_inline23927 = std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) - group_start_inline23926), 80);
                    uint32_t group_payload_inline23928_offsets[2] = {static_cast<uint32_t>(group_start_inline23926), 0};
                    uint32_t group_payload_inline23928_shapes[2] = {(group_payload_inline23928_offsets[0] >= proj_payload_inline23531.shapes[0] ? 0u : std::min<uint32_t>(80, proj_payload_inline23531.shapes[0] - group_payload_inline23928_offsets[0])), (group_payload_inline23928_offsets[1] >= proj_payload_inline23531.shapes[1] ? 0u : std::min<uint32_t>(5120, proj_payload_inline23531.shapes[1] - group_payload_inline23928_offsets[1]))};
                    Tensor group_payload_inline23928 = proj_payload_inline23531.view(group_payload_inline23928_shapes, group_payload_inline23928_offsets);
                    int64_t scale_start_inline23929 = (group_start_inline23926 * 160);
                    uint32_t group_scales_inline23930_offsets[2] = {0, static_cast<uint32_t>(scale_start_inline23929)};
                    uint32_t group_scales_inline23930_shapes[2] = {(group_scales_inline23930_offsets[0] >= proj_scales_inline23532.shapes[0] ? 0u : std::min<uint32_t>(1, proj_scales_inline23532.shapes[0] - group_scales_inline23930_offsets[0])), (group_scales_inline23930_offsets[1] >= proj_scales_inline23532.shapes[1] ? 0u : std::min<uint32_t>(12800, proj_scales_inline23532.shapes[1] - group_scales_inline23930_offsets[1]))};
                    Tensor group_scales_inline23930 = proj_scales_inline23532.view(group_scales_inline23930_shapes, group_scales_inline23930_offsets);

                    // Spmd proj_kv_2: _proj_kv
                    CoreTaskArgs params_t141;
                    params_t141.add_input(group_payload_inline23928);
                    params_t141.add_input(layer_wkv_inline23467);
                    params_t141.add_input(group_scales_inline23930);
                    params_t141.add_input(layer_wkv_scale_inline23468);
                    params_t141.add_inout(kv_projection_inline23925);
                    params_t141.add_scalar(group_rows_inline23927);
                    params_t141.add_scalar(group_start_inline23926);
                    params_t141.launch_spec.set_block_num(8);
                    params_t141.launch_spec.set_require_sync_start(true);
                    rt_submit_aic_task(15, params_t141);
                    uint32_t projection_group_inline23931_offsets[2] = {static_cast<uint32_t>(group_start_inline23926), 0};
                    uint32_t projection_group_inline23931_shapes[2] = {(projection_group_inline23931_offsets[0] >= kv_projection_inline23925.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline23927), kv_projection_inline23925.shapes[0] - projection_group_inline23931_offsets[0])), (projection_group_inline23931_offsets[1] >= kv_projection_inline23925.shapes[1] ? 0u : std::min<uint32_t>(512, kv_projection_inline23925.shapes[1] - projection_group_inline23931_offsets[1]))};
                    Tensor projection_group_inline23931 = kv_projection_inline23925.view(projection_group_inline23931_shapes, projection_group_inline23931_offsets);
                    uint32_t normalized_group_inline23932_offsets[2] = {static_cast<uint32_t>(group_start_inline23926), 0};
                    uint32_t normalized_group_inline23932_shapes[2] = {(normalized_group_inline23932_offsets[0] >= kv_inline23528__rv_v2.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline23927), kv_inline23528__rv_v2.shapes[0] - normalized_group_inline23932_offsets[0])), (normalized_group_inline23932_offsets[1] >= kv_inline23528__rv_v2.shapes[1] ? 0u : std::min<uint32_t>(512, kv_inline23528__rv_v2.shapes[1] - normalized_group_inline23932_offsets[1]))};
                    Tensor normalized_group_inline23932 = kv_inline23528__rv_v2.view(normalized_group_inline23932_shapes, normalized_group_inline23932_offsets);
                    int32_t group_tokens_inline23933 = static_cast<int32_t>(group_rows_inline23927);
                    int64_t spmd_core_num_15 = ((static_cast<int64_t>(group_tokens_inline23933) + 7) / 8);
                    if ((0 < spmd_core_num_15)) {

                        // Spmd norm_kv_spmd_2: norm_kv
                        CoreTaskArgs params_t142;
                        params_t142.add_input(projection_group_inline23931);
                        params_t142.add_input(layer_kv_norm_inline23469);
                        params_t142.add_inout(normalized_group_inline23932);
                        params_t142.add_scalar(group_tokens_inline23933);
                        params_t142.launch_spec.set_block_num(spmd_core_num_15);
                        rt_submit_aiv_task(16, params_t142);
                    } else {
                    }
                    int32_t t__tmp_v791 = rt_available_cluster_count();
                    int64_t rope_workers_inline23944 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v791) * 2), 5);
                    Tensor kv_inline23528__rv_v4 = ext_engram_lookup;
                    if ((0 < rope_workers_inline23944)) {

                        // Spmd rope_kv_spmd_2: rope_kv
                        CoreTaskArgs params_t143;
                        params_t143.add_inout(kv_inline23528__rv_v2);
                        params_t143.add_input(compressed_token_rope_cos_inline23437);
                        params_t143.add_input(compressed_token_rope_sin_inline23438);
                        params_t143.add_scalar(rope_workers_inline23944);
                        params_t143.add_scalar(group_start_inline23926);
                        params_t143.add_scalar(group_tokens_inline23933);
                        params_t143.launch_spec.set_block_num(rope_workers_inline23944);
                        rt_submit_aiv_task(17, params_t143);
                        kv_inline23528__rv_v4 = kv_inline23528__rv_v2;
                    } else {
                        kv_inline23528__rv_v4 = kv_inline23528__rv_v2;
                    }
                    kv_inline23528__rv_v2 = kv_inline23528__rv_v4;
                }
                int64_t workers_inline23978 = std::min<int64_t>(static_cast<int64_t>(attention_count_inline23434), 12);
                int64_t step_inline23979 = std::max<int64_t>(workers_inline23978, 1);
                if ((0 < workers_inline23978)) {

                    // Spmd pub_win_spmd_2: pub_win_2
                    CoreTaskArgs params_t144;
                    params_t144.add_inout(layer_window_cache_inline23453);
                    params_t144.add_inout(layer_window_cache_scale_inline23454);
                    params_t144.add_input(ext_window_slots);
                    params_t144.add_input(kv_inline23528__rv_v2);
                    params_t144.add_scalar(attention_count_inline23434);
                    params_t144.add_scalar(step_inline23979);
                    params_t144.launch_spec.set_block_num(workers_inline23978);
                    rt_submit_aiv_task(95, params_t144);
                } else {
                }
                int64_t blocks_inline24004 = (((static_cast<int64_t>(attention_count_inline23434) + 47) / 48) * 8);
                int64_t workers_inline24005 = std::max<int64_t>(blocks_inline24004, 1);

                // Spmd l2_decode_fwd_proj_cmp_0: l2_decode_fwd_incore_120
                CoreTaskArgs params_t145;
                params_t145.add_input(x_norm_inline23524__rv_v2);
                params_t145.add_input(ext_c1a_compressor_wkv);
                params_t145.add_inout(projected_inline23536);
                params_t145.add_scalar(attention_count_inline23434);
                params_t145.launch_spec.set_block_num(workers_inline24005);
                TaskId params_t145_deps[1];
                uint32_t params_t145_deps_count = 0;
                if (ret__tmp_v0_guarded_1_20.is_valid()) params_t145_deps[params_t145_deps_count++] = ret__tmp_v0_guarded_1_20;
                params_t145.set_dependencies(params_t145_deps, params_t145_deps_count);
                TaskOutputTensors task_145_outs = rt_submit_aic_task(96, params_t145);
                TaskId completed_inline24006 = task_145_outs.task_id();
                TaskId pc_tid_inline23538 = completed_inline24006;
                int64_t spmd_core_num_16 = ((static_cast<int64_t>(attention_count_inline23434) + 7) / 8);
                Tensor ret__tmp_v0_guarded_0_30 = ext_engram_lookup;
                TaskId ret__tmp_v0_guarded_1_22 = TaskId::invalid();
                if ((0 < spmd_core_num_16)) {

                    // Spmd norm_cmp_spmd: norm_cmp
                    CoreTaskArgs params_t146;
                    params_t146.add_input(projected_inline23536);
                    params_t146.add_input(full_compressor_norm_inline23479);
                    params_t146.add_inout(latent_inline23539);
                    params_t146.add_scalar(attention_count_inline23434);
                    params_t146.launch_spec.set_block_num(spmd_core_num_16);
                    TaskId params_t146_deps[1];
                    uint32_t params_t146_deps_count = 0;
                    params_t146_deps[params_t146_deps_count++] = completed_inline24006;
                    params_t146.set_dependencies(params_t146_deps, params_t146_deps_count);
                    TaskOutputTensors task_146_outs = rt_submit_aiv_task(97, params_t146);
                    TaskId ret__tmp_v0_nonempty_1_24 = task_146_outs.task_id();
                    ret__tmp_v0_guarded_0_30 = latent_inline23539;
                    ret__tmp_v0_guarded_1_22 = ret__tmp_v0_nonempty_1_24;
                } else {

                    // Phase-fence barrier 28: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_28;
                    TaskId params_phase_fence_barrier_28_deps[1];
                    uint32_t params_phase_fence_barrier_28_deps_count = 0;
                    params_phase_fence_barrier_28_deps[params_phase_fence_barrier_28_deps_count++] = completed_inline24006;
                    params_phase_fence_barrier_28.set_dependencies(params_phase_fence_barrier_28_deps, params_phase_fence_barrier_28_deps_count);
                    TaskId spmd_empty_tid_23 = TaskId::invalid();
                    if (params_phase_fence_barrier_28_deps_count > 0) {
                        TaskOutputTensors phase_fence_barrier_28_outs = rt_submit_dummy_task(params_phase_fence_barrier_28);
                        spmd_empty_tid_23 = phase_fence_barrier_28_outs.task_id();
                    }
                    ret__tmp_v0_guarded_0_30 = latent_inline23539;
                    ret__tmp_v0_guarded_1_22 = spmd_empty_tid_23;
                }
                Tensor latent_inline23539__ssa_v1 = ret__tmp_v0_guarded_0_30;
                TaskId nc_tid_inline23540 = ret__tmp_v0_guarded_1_22;
                int64_t blocks_inline24028 = ((static_cast<int64_t>(attention_count_inline23434) + 15) / 16);
                int64_t task_workers_inline24029 = std::max<int64_t>(std::min<int64_t>(blocks_inline24028, 8), 1);
                TaskId entry_deps_inline24030[1]; for (int64_t __init_i = 0; __init_i < 1; ++__init_i) entry_deps_inline24030[__init_i] = TaskId::invalid();
                entry_deps_inline24030[0] = nc_tid_inline23540;
                TaskId _submit_deps_buf_inline24031[1]; for (int64_t __init_i = 0; __init_i < 1; ++__init_i) _submit_deps_buf_inline24031[__init_i] = TaskId::invalid();
                TaskId t__tmp_v804 = entry_deps_inline24030[0];
                _submit_deps_buf_inline24031[0] = t__tmp_v804;

                // Spmd proj_idx_k_spmd_0: proj_idx_k
                CoreTaskArgs params_t147;
                params_t147.add_inout(projected_key_inline23542);
                params_t147.add_input(latent_inline23539__ssa_v1);
                params_t147.add_input(full_index_wk_inline23480);
                params_t147.add_scalar(blocks_inline24028);
                params_t147.add_scalar(task_workers_inline24029);
                params_t147.add_scalar(attention_count_inline23434);
                params_t147.launch_spec.set_block_num(task_workers_inline24029);
                TaskId params_t147_deps[1];
                uint32_t params_t147_deps_count = 0;
                if (_submit_deps_buf_inline24031[0].is_valid()) params_t147_deps[params_t147_deps_count++] = _submit_deps_buf_inline24031[0];
                params_t147.set_dependencies(params_t147_deps, params_t147_deps_count);
                TaskOutputTensors task_147_outs = rt_submit_aic_task(41, params_t147);
                TaskId cube_tid_inline24033 = task_147_outs.task_id();
                TaskId completed_inline24027 = cube_tid_inline24033;
                TaskId pk_tid_inline23543 = completed_inline24027;
                int64_t spmd_core_num_17 = ((static_cast<int64_t>(attention_count_inline23434) + 7) / 8);
                Tensor ret__tmp_v0_guarded_0_31 = ext_engram_lookup;
                TaskId ret__tmp_v0_guarded_1_23 = TaskId::invalid();
                if ((0 < spmd_core_num_17)) {

                    // Spmd norm_idx_k_spmd_0: norm_idx_k_0
                    CoreTaskArgs params_t148;
                    params_t148.add_input(projected_key_inline23542);
                    params_t148.add_input(full_index_norm_inline23481);
                    params_t148.add_inout(normalized_key_inline23544);
                    params_t148.add_scalar(attention_count_inline23434);
                    params_t148.launch_spec.set_block_num(spmd_core_num_17);
                    TaskId params_t148_deps[1];
                    uint32_t params_t148_deps_count = 0;
                    params_t148_deps[params_t148_deps_count++] = cube_tid_inline24033;
                    params_t148.set_dependencies(params_t148_deps, params_t148_deps_count);
                    TaskOutputTensors task_148_outs = rt_submit_aiv_task(98, params_t148);
                    TaskId ret__tmp_v0_nonempty_1_25 = task_148_outs.task_id();
                    ret__tmp_v0_guarded_0_31 = normalized_key_inline23544;
                    ret__tmp_v0_guarded_1_23 = ret__tmp_v0_nonempty_1_25;
                } else {

                    // Phase-fence barrier 29: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_29;
                    TaskId params_phase_fence_barrier_29_deps[1];
                    uint32_t params_phase_fence_barrier_29_deps_count = 0;
                    params_phase_fence_barrier_29_deps[params_phase_fence_barrier_29_deps_count++] = cube_tid_inline24033;
                    params_phase_fence_barrier_29.set_dependencies(params_phase_fence_barrier_29_deps, params_phase_fence_barrier_29_deps_count);
                    TaskId spmd_empty_tid_24 = TaskId::invalid();
                    if (params_phase_fence_barrier_29_deps_count > 0) {
                        TaskOutputTensors phase_fence_barrier_29_outs = rt_submit_dummy_task(params_phase_fence_barrier_29);
                        spmd_empty_tid_24 = phase_fence_barrier_29_outs.task_id();
                    }
                    ret__tmp_v0_guarded_0_31 = normalized_key_inline23544;
                    ret__tmp_v0_guarded_1_23 = spmd_empty_tid_24;
                }
                Tensor normalized_key_inline23544__ssa_v1 = ret__tmp_v0_guarded_0_31;
                TaskId nk_tid_inline23545 = ret__tmp_v0_guarded_1_23;
                int32_t blocks_inline24052 = attention_count_inline23434;
                int64_t workers_inline24053 = std::min<int64_t>(static_cast<int64_t>(blocks_inline24052), 32);
                Tensor ret__tmp_v0_guarded_0_32 = ext_engram_lookup;
                TaskId ret__tmp_v0_guarded_1_24 = TaskId::invalid();
                if ((0 < workers_inline24053)) {

                    // Spmd rope_idx_k_spmd_0: rope_idx_k
                    CoreTaskArgs params_t149;
                    params_t149.add_inout(index_key_inline23547);
                    params_t149.add_input(normalized_key_inline23544__ssa_v1);
                    params_t149.add_input(compressed_token_rope_cos_inline23437);
                    params_t149.add_input(compressed_token_rope_sin_inline23438);
                    params_t149.add_scalar(blocks_inline24052);
                    params_t149.add_scalar(workers_inline24053);
                    params_t149.launch_spec.set_block_num(workers_inline24053);
                    TaskId params_t149_deps[1];
                    uint32_t params_t149_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_23.is_valid()) params_t149_deps[params_t149_deps_count++] = ret__tmp_v0_guarded_1_23;
                    params_t149.set_dependencies(params_t149_deps, params_t149_deps_count);
                    TaskOutputTensors task_149_outs = rt_submit_aiv_task(43, params_t149);
                    TaskId ret__tmp_v0_nonempty_1_26 = task_149_outs.task_id();
                    ret__tmp_v0_guarded_0_32 = index_key_inline23547;
                    ret__tmp_v0_guarded_1_24 = ret__tmp_v0_nonempty_1_26;
                } else {

                    // Phase-fence barrier 30: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_30;
                    TaskId params_phase_fence_barrier_30_deps[1];
                    uint32_t params_phase_fence_barrier_30_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_23.is_valid()) params_phase_fence_barrier_30_deps[params_phase_fence_barrier_30_deps_count++] = ret__tmp_v0_guarded_1_23;
                    params_phase_fence_barrier_30.set_dependencies(params_phase_fence_barrier_30_deps, params_phase_fence_barrier_30_deps_count);
                    TaskId spmd_empty_tid_25 = TaskId::invalid();
                    if (params_phase_fence_barrier_30_deps_count > 0) {
                        TaskOutputTensors phase_fence_barrier_30_outs = rt_submit_dummy_task(params_phase_fence_barrier_30);
                        spmd_empty_tid_25 = phase_fence_barrier_30_outs.task_id();
                    }
                    ret__tmp_v0_guarded_0_32 = index_key_inline23547;
                    ret__tmp_v0_guarded_1_24 = spmd_empty_tid_25;
                }
                Tensor index_key_inline23547__rv_v2 = ret__tmp_v0_guarded_0_32;
                TaskId rope_tid_inline24054 = ret__tmp_v0_guarded_1_24;
                TaskId ik_tid_inline23548 = rope_tid_inline24054;
                int64_t t__tmp_v823 = (int64_t)full_index_cache_inline23476.shapes[0];
                int64_t cache_rows_inline24069 = (t__tmp_v823 * 128);
                uint32_t flat_inline24070_shapes[2] = {static_cast<uint32_t>(cache_rows_inline24069), 64};
                Tensor flat_inline24070 = full_index_cache_inline23476.reshape(flat_inline24070_shapes, 2);
                uint32_t flat_scale_inline24071_shapes[2] = {static_cast<uint32_t>(cache_rows_inline24069), 4};
                Tensor flat_scale_inline24071 = full_index_cache_scale_inline23477.reshape(flat_scale_inline24071_shapes, 2);
                int64_t workers_inline24072 = std::min<int64_t>(static_cast<int64_t>(attention_count_inline23434), 4);
                TaskId ret__tmp_v0_guarded_0_33 = TaskId::invalid();
                if ((0 < workers_inline24072)) {

                    // Spmd pub_idx_spmd_0: pub_idx
                    CoreTaskArgs params_t150;
                    params_t150.add_input(ext_c1a_compressed_slots);
                    params_t150.add_input(index_key_inline23547__rv_v2);
                    params_t150.add_inout(flat_inline24070);
                    params_t150.add_inout(flat_scale_inline24071);
                    params_t150.add_scalar(attention_count_inline23434);
                    params_t150.add_scalar(workers_inline24072);
                    params_t150.launch_spec.set_block_num(workers_inline24072);
                    TaskId params_t150_deps[2];
                    uint32_t params_t150_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_20.is_valid()) params_t150_deps[params_t150_deps_count++] = ret__tmp_v0_guarded_1_20;
                    if (ret__tmp_v0_guarded_1_24.is_valid()) params_t150_deps[params_t150_deps_count++] = ret__tmp_v0_guarded_1_24;
                    params_t150.set_dependencies(params_t150_deps, params_t150_deps_count);
                    TaskOutputTensors task_150_outs = rt_submit_aiv_task(44, params_t150);
                    TaskId ret__tmp_v0_nonempty_0_37 = task_150_outs.task_id();
                    ret__tmp_v0_guarded_0_33 = ret__tmp_v0_nonempty_0_37;
                } else {

                    // Phase-fence barrier 31: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_31;
                    TaskId params_phase_fence_barrier_31_deps[2];
                    uint32_t params_phase_fence_barrier_31_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_20.is_valid()) params_phase_fence_barrier_31_deps[params_phase_fence_barrier_31_deps_count++] = ret__tmp_v0_guarded_1_20;
                    if (ret__tmp_v0_guarded_1_24.is_valid()) params_phase_fence_barrier_31_deps[params_phase_fence_barrier_31_deps_count++] = ret__tmp_v0_guarded_1_24;
                    params_phase_fence_barrier_31.set_dependencies(params_phase_fence_barrier_31_deps, params_phase_fence_barrier_31_deps_count);
                    TaskId spmd_empty_tid_26 = TaskId::invalid();
                    if (params_phase_fence_barrier_31_deps_count > 0) {
                        TaskOutputTensors phase_fence_barrier_31_outs = rt_submit_dummy_task(params_phase_fence_barrier_31);
                        spmd_empty_tid_26 = phase_fence_barrier_31_outs.task_id();
                    }
                    ret__tmp_v0_guarded_0_33 = spmd_empty_tid_26;
                }
                TaskId published_inline24073 = ret__tmp_v0_guarded_0_33;
                TaskId index_ready_inline23549 = published_inline24073;
                int32_t blocks_inline24138 = attention_count_inline23434;
                int64_t workers_inline24139 = std::min<int64_t>(static_cast<int64_t>(blocks_inline24138), 12);
                Tensor compressed_rotated_inline23550__ssa_v1 = ext_engram_lookup;
                if ((0 < workers_inline24139)) {

                    // Spmd rope_cmp_spmd: rope_cmp
                    CoreTaskArgs params_t151;
                    params_t151.add_input(latent_inline23539__ssa_v1);
                    params_t151.add_input(compressed_token_rope_cos_inline23437);
                    params_t151.add_input(compressed_token_rope_sin_inline23438);
                    params_t151.add_inout(compressed_rotated_inline23550);
                    params_t151.add_scalar(blocks_inline24138);
                    params_t151.add_scalar(workers_inline24139);
                    params_t151.launch_spec.set_block_num(workers_inline24139);
                    rt_submit_aiv_task(99, params_t151);
                    compressed_rotated_inline23550__ssa_v1 = compressed_rotated_inline23550;
                } else {
                    compressed_rotated_inline23550__ssa_v1 = compressed_rotated_inline23550;
                }
                Tensor compressed_rotated_v1_inline23551 = compressed_rotated_inline23550__ssa_v1;
                int64_t t__tmp_v844 = (int64_t)layer_compressed_cache_inline23456.shapes[0];
                int64_t cache_rows_inline24167 = (t__tmp_v844 * 128);
                uint32_t flat_inline24168_shapes[2] = {static_cast<uint32_t>(cache_rows_inline24167), 256};
                Tensor flat_inline24168 = layer_compressed_cache_inline23456.reshape(flat_inline24168_shapes, 2);
                uint32_t flat_scale_inline24169_shapes[2] = {static_cast<uint32_t>(cache_rows_inline24167), 32};
                Tensor flat_scale_inline24169 = layer_compressed_cache_scale_inline23457.reshape(flat_scale_inline24169_shapes, 2);
                int64_t workers_inline24170 = std::min<int64_t>(static_cast<int64_t>(attention_count_inline23434), 8);
                TaskId ret__tmp_v0_guarded_0_34 = TaskId::invalid();
                if ((0 < workers_inline24170)) {

                    // Spmd pub_cmp_spmd: pub_cmp
                    CoreTaskArgs params_t152;
                    params_t152.add_input(ext_c1a_compressed_slots);
                    params_t152.add_input(compressed_rotated_v1_inline23551);
                    params_t152.add_inout(flat_inline24168);
                    params_t152.add_inout(flat_scale_inline24169);
                    params_t152.add_scalar(attention_count_inline23434);
                    params_t152.add_scalar(workers_inline24170);
                    params_t152.launch_spec.set_block_num(workers_inline24170);
                    TaskId params_t152_deps[1];
                    uint32_t params_t152_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_20.is_valid()) params_t152_deps[params_t152_deps_count++] = ret__tmp_v0_guarded_1_20;
                    params_t152.set_dependencies(params_t152_deps, params_t152_deps_count);
                    TaskOutputTensors task_152_outs = rt_submit_aiv_task(100, params_t152);
                    TaskId ret__tmp_v0_nonempty_0_38 = task_152_outs.task_id();
                    ret__tmp_v0_guarded_0_34 = ret__tmp_v0_nonempty_0_38;
                } else {

                    // Phase-fence barrier 32: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_32;
                    TaskId params_phase_fence_barrier_32_deps[1];
                    uint32_t params_phase_fence_barrier_32_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_20.is_valid()) params_phase_fence_barrier_32_deps[params_phase_fence_barrier_32_deps_count++] = ret__tmp_v0_guarded_1_20;
                    params_phase_fence_barrier_32.set_dependencies(params_phase_fence_barrier_32_deps, params_phase_fence_barrier_32_deps_count);
                    TaskId spmd_empty_tid_27 = TaskId::invalid();
                    if (params_phase_fence_barrier_32_deps_count > 0) {
                        TaskOutputTensors phase_fence_barrier_32_outs = rt_submit_dummy_task(params_phase_fence_barrier_32);
                        spmd_empty_tid_27 = phase_fence_barrier_32_outs.task_id();
                    }
                    ret__tmp_v0_guarded_0_34 = spmd_empty_tid_27;
                }
                TaskId published_inline24171 = ret__tmp_v0_guarded_0_34;
                TaskId compressed_ready_inline23552 = published_inline24171;
                int64_t score_width_inline23553 = (FWD_C1A_TABLE_DYN * 128);
                for (int64_t group_start_inline24223 = 0; group_start_inline24223 < attention_count_inline23434; group_start_inline24223 += 80) {
                    int64_t rows_inline24224 = std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) - group_start_inline24223), 80);
                    int32_t t__tmp_v862 = rt_available_cluster_count();
                    int64_t query_workers_inline24225 = std::min<int64_t>(static_cast<int64_t>(t__tmp_v862), 16);
                    int64_t query_workers_v1_inline24226 = std::max<int64_t>(query_workers_inline24225, 1);

                    // Spmd proj_idx_q_0: _proj_idx_q
                    CoreTaskArgs params_t153;
                    params_t153.add_input(qb_payload_inline23534);
                    params_t153.add_input(full_index_wq_b_inline23482);
                    params_t153.add_input(qb_scales_inline23535);
                    params_t153.add_input(full_index_wq_b_scale_inline23478);
                    params_t153.add_inout(projected_inline24222);
                    params_t153.add_scalar(rows_inline24224);
                    params_t153.add_scalar((uint64_t)4096);
                    params_t153.add_scalar(group_start_inline24223);
                    params_t153.add_scalar(group_start_inline24223);
                    params_t153.launch_spec.set_block_num(query_workers_v1_inline24226);
                    rt_submit_aic_task(45, params_t153);
                }
                TaskId entry_deps_inline24233[1]; for (int64_t __init_i = 0; __init_i < 1; ++__init_i) entry_deps_inline24233[__init_i] = TaskId::invalid();
                entry_deps_inline24233[0] = x_ready_inline23525;
                TaskId _submit_deps_buf_inline24234[1]; for (int64_t __init_i = 0; __init_i < 1; ++__init_i) _submit_deps_buf_inline24234[__init_i] = TaskId::invalid();
                TaskId t__tmp_v863 = entry_deps_inline24233[0];
                _submit_deps_buf_inline24234[0] = t__tmp_v863;

                // Spmd idx_q_fp4_spmd_0: idx_q_fp4
                CoreTaskArgs params_t154;
                params_t154.add_input(projected_inline24222);
                params_t154.add_input(compressed_token_rope_cos_inline23437);
                params_t154.add_input(compressed_token_rope_sin_inline23438);
                params_t154.add_inout(query_inline24227);
                params_t154.add_inout(query_scale_inline24228);
                params_t154.add_scalar(attention_count_inline23434);
                params_t154.launch_spec.set_block_num(32);
                TaskId params_t154_deps[1];
                uint32_t params_t154_deps_count = 0;
                if (_submit_deps_buf_inline24234[0].is_valid()) params_t154_deps[params_t154_deps_count++] = _submit_deps_buf_inline24234[0];
                params_t154.set_dependencies(params_t154_deps, params_t154_deps_count);
                TaskOutputTensors task_154_outs = rt_submit_aiv_task(46, params_t154);
                TaskId quant_tid_inline24236 = task_154_outs.task_id();
                TaskId query_tid_inline24229 = quant_tid_inline24236;
                int64_t blocks_inline24307 = ((static_cast<int64_t>(attention_count_inline23434) + 79) / 80);

                // Spmd proj_idx_w_spmd_0: proj_idx_w
                CoreTaskArgs params_t155;
                params_t155.add_inout(partials_inline24306);
                params_t155.add_input(x_norm_inline23524__rv_v2);
                params_t155.add_input(full_index_weights_proj_inline23483);
                params_t155.add_scalar(LOCAL_T_DYN);
                params_t155.add_scalar(blocks_inline24307);
                params_t155.add_scalar(attention_count_inline23434);
                params_t155.launch_spec.set_block_num(4);
                TaskId params_t155_deps[1];
                uint32_t params_t155_deps_count = 0;
                if (ret__tmp_v0_guarded_1_20.is_valid()) params_t155_deps[params_t155_deps_count++] = ret__tmp_v0_guarded_1_20;
                params_t155.set_dependencies(params_t155_deps, params_t155_deps_count);
                TaskOutputTensors task_155_outs = rt_submit_aic_task(47, params_t155);
                TaskId cube_tid_inline24308 = task_155_outs.task_id();

                // Spmd reduce_idx_w_spmd_0: reduce_idx_w
                CoreTaskArgs params_t156;
                params_t156.add_inout(weights_inline24230);
                params_t156.add_input(partials_inline24306);
                params_t156.add_scalar(LOCAL_T_DYN);
                params_t156.add_scalar(attention_count_inline23434);
                params_t156.launch_spec.set_block_num(std::max<int64_t>(std::min<int64_t>(((static_cast<int64_t>(attention_count_inline23434) + 15) / 16), 8), 1));
                TaskId params_t156_deps[1];
                uint32_t params_t156_deps_count = 0;
                params_t156_deps[params_t156_deps_count++] = cube_tid_inline24308;
                params_t156.set_dependencies(params_t156_deps, params_t156_deps_count);
                TaskOutputTensors task_156_outs = rt_submit_aiv_task(48, params_t156);
                TaskId completed_inline24322 = task_156_outs.task_id();
                TaskId weights_tid_inline24231 = completed_inline24322;
                uint32_t scores_inline24232_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), static_cast<uint32_t>(score_width_inline23553)};
                TensorCreateInfo scores_inline24232_ci(scores_inline24232_ci_shapes, 2, DataType::FLOAT32);
                TaskOutputTensors alloc_62 = alloc_tensors(scores_inline24232_ci);
                const Tensor& scores_inline24232 = alloc_62.get_ref(0);
                Tensor index_query_inline23555 = query_inline24227;
                Tensor index_query_scale_inline23556 = query_scale_inline24228;
                Tensor index_weights_inline23557 = weights_inline24230;
                Tensor scores_inline23558 = scores_inline24232;
                TaskId query_tid_inline23559 = query_tid_inline24229;
                TaskId weights_tid_inline23560 = weights_tid_inline24231;
                int64_t width_inline24330 = (int64_t)scores_inline23558.shapes[1];

                // Task 157: idx_plan_req_0
                CoreTaskArgs params_t157;
                params_t157.add_input(ext_token_to_req_indices);
                params_t157.add_inout(runs_inline24332);
                params_t157.add_input(c1a_compressed_lens_inline23439);
                params_t157.add_scalar(attention_count_inline23434);
                params_t157.add_scalar(width_inline24330);
                TaskOutputTensors task_157_outs = rt_submit_aiv_task(101, params_t157);
                TaskId plan_tid_inline24333 = task_157_outs.task_id();
                int32_t t__tmp_v887 = rt_available_cluster_count();
                int64_t launch_workers_inline24341 = std::max<int64_t>(std::min<int64_t>(static_cast<int64_t>(t__tmp_v887), 32), 1);
                int64_t run_stride_inline24342 = (int64_t)runs_inline24332.shapes[1];

                // Group _request_index_scores: MixedKernels (AIC + AIV lanes)
                CoreTaskArgs params_t158;
                params_t158.add_input(index_query_inline23555);
                params_t158.add_input(index_query_scale_inline23556);
                params_t158.add_input(index_weights_inline23557);
                params_t158.add_input(full_index_cache_inline23476);
                params_t158.add_input(full_index_cache_scale_inline23477);
                params_t158.add_input(ext_c1a_index_block_table);
                params_t158.add_input(ext_token_to_req_indices);
                params_t158.add_input(c1a_compressed_lens_inline23439);
                params_t158.add_inout(scores_inline23558);
                params_t158.add_input(runs_inline24332);
                params_t158.add_input(ext_c1a_index_block_table);
                params_t158.add_scalar(width_inline24330);
                params_t158.add_scalar(run_stride_inline24342);
                params_t158.add_scalar(FWD_C1A_TABLE_DYN);
                params_t158.add_scalar((uint64_t)static_cast<int32_t>(0));
                params_t158.add_scalar((uint64_t)0);
                params_t158.add_scalar((uint64_t)static_cast<int32_t>(0));
                MixedKernels mixed_158 = {50, 51, INVALID_KERNEL_ID};
                params_t158.launch_spec.set_block_num(launch_workers_inline24341);
                params_t158.launch_spec.set_require_sync_start(true);
                TaskId params_t158_deps[4];
                uint32_t params_t158_deps_count = 0;
                if (ret__tmp_v0_guarded_0_33.is_valid()) params_t158_deps[params_t158_deps_count++] = ret__tmp_v0_guarded_0_33;
                params_t158_deps[params_t158_deps_count++] = quant_tid_inline24236;
                params_t158_deps[params_t158_deps_count++] = completed_inline24322;
                params_t158_deps[params_t158_deps_count++] = plan_tid_inline24333;
                params_t158.set_dependencies(params_t158_deps, params_t158_deps_count);
                TaskOutputTensors task_158_outs = rt_submit_task(mixed_158, params_t158);
                TaskId score_tid_inline24344 = task_158_outs.task_id();
                TaskId scores_tid_inline23561 = score_tid_inline24344;
                int64_t width_inline24345 = (int64_t)scores_inline23558.shapes[1];
                TaskId topk_tid_inline24347 = TaskId::invalid();
                Tensor topk_indices_inline23445__rv_v1 = ext_engram_lookup;
                if ((width_inline24345 <= 512)) {

                    // Spmd idx_topk_short_spmd_0: idx_topk_short_0
                    CoreTaskArgs params_t159;
                    params_t159.add_input(ext_token_to_req_indices);
                    params_t159.add_input(ext_c1a_index_block_table);
                    params_t159.add_inout(topk_indices_inline23445__rv_v5);
                    params_t159.add_input(scores_inline23558);
                    params_t159.add_scalar(FWD_C1A_TABLE_DYN);
                    params_t159.add_scalar(attention_count_inline23434);
                    params_t159.add_scalar(width_inline24345);
                    params_t159.launch_spec.set_block_num(32);
                    TaskId params_t159_deps[1];
                    uint32_t params_t159_deps_count = 0;
                    params_t159_deps[params_t159_deps_count++] = score_tid_inline24344;
                    params_t159.set_dependencies(params_t159_deps, params_t159_deps_count);
                    TaskOutputTensors task_159_outs = rt_submit_aiv_task(102, params_t159);
                    TaskId short_tid_inline24348 = task_159_outs.task_id();
                    topk_tid_inline24347 = short_tid_inline24348;
                    topk_indices_inline23445__rv_v1 = topk_indices_inline23445__rv_v5;
                } else {
                    int64_t leaf_count_inline24353 = ((width_inline24345 + 2047) / 2048);
                    int64_t arena_rows_inline24354 = (LOCAL_T_DYN * leaf_count_inline24353);
                    uint32_t pair_arena_inline24355_ci_shapes[2] = {static_cast<uint32_t>(arena_rows_inline24354), 1024};
                    TensorCreateInfo pair_arena_inline24355_ci(pair_arena_inline24355_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_63 = alloc_tensors(pair_arena_inline24355_ci);
                    const Tensor& pair_arena_inline24355 = alloc_63.get_ref(0);
                    int64_t total_leaves_inline24356 = (static_cast<int64_t>(attention_count_inline23434) * leaf_count_inline24353);

                    // Spmd idx_topk_leaf_spmd_0: idx_topk_leaf_0
                    CoreTaskArgs params_t160;
                    params_t160.add_input(scores_inline23558);
                    params_t160.add_inout(pair_arena_inline24355);
                    params_t160.add_scalar(total_leaves_inline24356);
                    params_t160.add_scalar(leaf_count_inline24353);
                    params_t160.add_scalar(width_inline24345);
                    params_t160.launch_spec.set_block_num(32);
                    TaskId params_t160_deps[1];
                    uint32_t params_t160_deps_count = 0;
                    params_t160_deps[params_t160_deps_count++] = score_tid_inline24344;
                    params_t160.set_dependencies(params_t160_deps, params_t160_deps_count);
                    TaskOutputTensors task_160_outs = rt_submit_aiv_task(103, params_t160);
                    TaskId leaf_tid_inline24357 = task_160_outs.task_id();

                    // Spmd idx_topk_merge_spmd_0: idx_topk_merge_0
                    CoreTaskArgs params_t161;
                    params_t161.add_inout(pair_arena_inline24355);
                    params_t161.add_input(ext_token_to_req_indices);
                    params_t161.add_input(ext_c1a_index_block_table);
                    params_t161.add_inout(topk_indices_inline23445__rv_v5);
                    params_t161.add_input(scores_inline23558);
                    params_t161.add_scalar(FWD_C1A_TABLE_DYN);
                    params_t161.add_scalar(attention_count_inline23434);
                    params_t161.add_scalar(leaf_count_inline24353);
                    params_t161.add_scalar(width_inline24345);
                    params_t161.launch_spec.set_block_num(32);
                    TaskId params_t161_deps[1];
                    uint32_t params_t161_deps_count = 0;
                    params_t161_deps[params_t161_deps_count++] = leaf_tid_inline24357;
                    params_t161.set_dependencies(params_t161_deps, params_t161_deps_count);
                    TaskOutputTensors task_161_outs = rt_submit_aiv_task(104, params_t161);
                    TaskId merge_tid_inline24365 = task_161_outs.task_id();
                    topk_tid_inline24347 = merge_tid_inline24365;
                    topk_indices_inline23445__rv_v1 = topk_indices_inline23445__rv_v5;
                }
                TaskId topk_tid_inline23562 = topk_tid_inline24347;
                int64_t positions_inline24528 = (int64_t)scores_inline23558.shapes[1];
                int64_t capacity_inline24529 = (int64_t)pool_block_ids_inline23448__rv_v3.shapes[1];
                int64_t active_tokens_inline24530 = static_cast<int64_t>(attention_count_inline23434);
                int64_t active_tokens_inline24530__phi_v2;
                if ((active_tokens_inline24530 < 0)) {
                    active_tokens_inline24530__phi_v2 = 0;
                } else {
                    active_tokens_inline24530__phi_v2 = active_tokens_inline24530;
                }
                int64_t active_tokens_inline24530__phi_v4;
                if ((LOCAL_T_DYN < active_tokens_inline24530__phi_v2)) {
                    int64_t active_tokens_inline24530__ssa_v3 = static_cast<int64_t>(LOCAL_T_DYN);
                    active_tokens_inline24530__phi_v4 = active_tokens_inline24530__ssa_v3;
                } else {
                    active_tokens_inline24530__phi_v4 = active_tokens_inline24530__phi_v2;
                }
                int64_t block_count_inline24531 = ((positions_inline24528 + 7) / 8);
                TaskId pool_tid_inline24532 = TaskId::invalid();
                Tensor pool_block_ids_inline23448__rv_v1 = ext_engram_lookup;
                if ((block_count_inline24531 <= std::min<int64_t>(capacity_inline24529, 2048))) {
                    int64_t all_workers_inline24533 = std::max<int64_t>(std::min<int64_t>(LOCAL_T_DYN, 12), 1);

                    // Spmd pool_all_spmd: pool_all
                    CoreTaskArgs params_t162;
                    params_t162.add_input(c1a_compressed_lens_inline23439);
                    params_t162.add_inout(pool_block_ids_inline23448__rv_v3);
                    params_t162.add_inout(pool_block_counts_inline23449);
                    params_t162.add_scalar(T_DYN);
                    params_t162.add_scalar(LOCAL_T_DYN);
                    params_t162.add_scalar(all_workers_inline24533);
                    params_t162.add_scalar(active_tokens_inline24530__phi_v4);
                    params_t162.add_scalar(positions_inline24528);
                    params_t162.add_scalar(capacity_inline24529);
                    params_t162.launch_spec.set_block_num(all_workers_inline24533);
                    TaskOutputTensors task_162_outs = rt_submit_aiv_task(105, params_t162);
                    TaskId all_tid_inline24534 = task_162_outs.task_id();
                    pool_tid_inline24532 = all_tid_inline24534;
                    pool_block_ids_inline23448__rv_v1 = pool_block_ids_inline23448__rv_v3;
                } else {
                    uint32_t arena_inline24565_ci_shapes[2] = {static_cast<uint32_t>((LOCAL_T_DYN * 16)), 4096};
                    TensorCreateInfo arena_inline24565_ci(arena_inline24565_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_64 = alloc_tensors(arena_inline24565_ci);
                    const Tensor& arena_inline24565 = alloc_64.get_ref(0);
                    int64_t padded_block_count_inline24542 = (((block_count_inline24531 + 8191) / 8192) * 8192);
                    int64_t score_tiles_inline24543 = ((block_count_inline24531 + 1023) / 1024);
                    uint32_t block_scores_inline24544_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), static_cast<uint32_t>(padded_block_count_inline24542)};
                    TensorCreateInfo block_scores_inline24544_ci(block_scores_inline24544_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_65 = alloc_tensors(block_scores_inline24544_ci);
                    const Tensor& block_scores_inline24544 = alloc_65.get_ref(0);
                    int64_t score_task_tokens_inline24545 = std::max<int64_t>(active_tokens_inline24530__phi_v4, 1);
                    int64_t score_work_items_inline24546 = (score_task_tokens_inline24545 * score_tiles_inline24543);
                    int64_t score_workers_inline24547 = std::min<int64_t>(score_work_items_inline24546, 32);
                    Tensor ret__tmp_v0_guarded_0_35 = ext_engram_lookup;
                    TaskId ret__tmp_v0_guarded_1_25 = TaskId::invalid();
                    if ((0 < score_workers_inline24547)) {

                        // Spmd pool_score_spmd: pool_score
                        CoreTaskArgs params_t163;
                        params_t163.add_input(scores_inline23558);
                        params_t163.add_input(c1a_compressed_lens_inline23439);
                        params_t163.add_inout(block_scores_inline24544);
                        params_t163.add_scalar(score_work_items_inline24546);
                        params_t163.add_scalar(score_workers_inline24547);
                        params_t163.add_scalar(score_tiles_inline24543);
                        params_t163.add_scalar(active_tokens_inline24530__phi_v4);
                        params_t163.add_scalar(block_count_inline24531);
                        params_t163.add_scalar(positions_inline24528);
                        params_t163.launch_spec.set_block_num(score_workers_inline24547);
                        TaskId params_t163_deps[1];
                        uint32_t params_t163_deps_count = 0;
                        params_t163_deps[params_t163_deps_count++] = score_tid_inline24344;
                        params_t163.set_dependencies(params_t163_deps, params_t163_deps_count);
                        TaskOutputTensors task_163_outs = rt_submit_aiv_task(106, params_t163);
                        TaskId ret__tmp_v0_nonempty_1_27 = task_163_outs.task_id();
                        ret__tmp_v0_guarded_0_35 = block_scores_inline24544;
                        ret__tmp_v0_guarded_1_25 = ret__tmp_v0_nonempty_1_27;
                    } else {

                        // Phase-fence barrier 33: dependency-only dummy task
                        CoreTaskArgs params_phase_fence_barrier_33;
                        TaskId params_phase_fence_barrier_33_deps[1];
                        uint32_t params_phase_fence_barrier_33_deps_count = 0;
                        params_phase_fence_barrier_33_deps[params_phase_fence_barrier_33_deps_count++] = score_tid_inline24344;
                        params_phase_fence_barrier_33.set_dependencies(params_phase_fence_barrier_33_deps, params_phase_fence_barrier_33_deps_count);
                        TaskId spmd_empty_tid_28 = TaskId::invalid();
                        if (params_phase_fence_barrier_33_deps_count > 0) {
                            TaskOutputTensors phase_fence_barrier_33_outs = rt_submit_dummy_task(params_phase_fence_barrier_33);
                            spmd_empty_tid_28 = phase_fence_barrier_33_outs.task_id();
                        }
                        ret__tmp_v0_guarded_0_35 = block_scores_inline24544;
                        ret__tmp_v0_guarded_1_25 = spmd_empty_tid_28;
                    }
                    Tensor block_scores_inline24544__ssa_v1 = ret__tmp_v0_guarded_0_35;
                    TaskId score_tid_inline24548 = ret__tmp_v0_guarded_1_25;
                    Tensor ret__tmp_v0_guarded_0_36 = ext_engram_lookup;
                    TaskId ret__tmp_v0_guarded_1_26 = TaskId::invalid();
                    if ((0 < LOCAL_T_DYN)) {

                        // Spmd pool_select_spmd: pool_select
                        CoreTaskArgs params_t164;
                        params_t164.add_input(c1a_compressed_lens_inline23439);
                        params_t164.add_input(block_scores_inline24544__ssa_v1);
                        params_t164.add_inout(arena_inline24565);
                        params_t164.add_inout(pool_block_ids_inline23448__rv_v3);
                        params_t164.add_scalar(active_tokens_inline24530__phi_v4);
                        params_t164.add_scalar(positions_inline24528);
                        params_t164.add_scalar(capacity_inline24529);
                        params_t164.launch_spec.set_block_num(LOCAL_T_DYN);
                        TaskId params_t164_deps[1];
                        uint32_t params_t164_deps_count = 0;
                        if (ret__tmp_v0_guarded_1_25.is_valid()) params_t164_deps[params_t164_deps_count++] = ret__tmp_v0_guarded_1_25;
                        params_t164.set_dependencies(params_t164_deps, params_t164_deps_count);
                        TaskOutputTensors task_164_outs = rt_submit_aiv_task(107, params_t164);
                        TaskId ret__tmp_v0_nonempty_1_28 = task_164_outs.task_id();
                        ret__tmp_v0_guarded_0_36 = pool_block_ids_inline23448__rv_v3;
                        ret__tmp_v0_guarded_1_26 = ret__tmp_v0_nonempty_1_28;
                    } else {

                        // Phase-fence barrier 34: dependency-only dummy task
                        CoreTaskArgs params_phase_fence_barrier_34;
                        TaskId params_phase_fence_barrier_34_deps[1];
                        uint32_t params_phase_fence_barrier_34_deps_count = 0;
                        if (ret__tmp_v0_guarded_1_25.is_valid()) params_phase_fence_barrier_34_deps[params_phase_fence_barrier_34_deps_count++] = ret__tmp_v0_guarded_1_25;
                        params_phase_fence_barrier_34.set_dependencies(params_phase_fence_barrier_34_deps, params_phase_fence_barrier_34_deps_count);
                        TaskId spmd_empty_tid_29 = TaskId::invalid();
                        if (params_phase_fence_barrier_34_deps_count > 0) {
                            TaskOutputTensors phase_fence_barrier_34_outs = rt_submit_dummy_task(params_phase_fence_barrier_34);
                            spmd_empty_tid_29 = phase_fence_barrier_34_outs.task_id();
                        }
                        ret__tmp_v0_guarded_0_36 = pool_block_ids_inline23448__rv_v3;
                        ret__tmp_v0_guarded_1_26 = spmd_empty_tid_29;
                    }
                    Tensor pool_block_ids_inline23448__ssa_v2 = ret__tmp_v0_guarded_0_36;
                    TaskId selected_tid_inline24566 = ret__tmp_v0_guarded_1_26;

                    // Task 165: pool_count
                    CoreTaskArgs params_t165;
                    params_t165.add_input(c1a_compressed_lens_inline23439);
                    params_t165.add_inout(pool_block_counts_inline23449);
                    params_t165.add_scalar(T_DYN);
                    params_t165.add_scalar(active_tokens_inline24530__phi_v4);
                    params_t165.add_scalar(positions_inline24528);
                    params_t165.add_scalar(capacity_inline24529);
                    TaskId params_t165_deps[1];
                    uint32_t params_t165_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_26.is_valid()) params_t165_deps[params_t165_deps_count++] = ret__tmp_v0_guarded_1_26;
                    params_t165.set_dependencies(params_t165_deps, params_t165_deps_count);
                    TaskOutputTensors task_165_outs = rt_submit_aiv_task(108, params_t165);
                    TaskId counts_tid_inline24585 = task_165_outs.task_id();
                    pool_tid_inline24532 = counts_tid_inline24585;
                    pool_block_ids_inline23448__rv_v1 = pool_block_ids_inline23448__ssa_v2;
                }
                TaskId pool_tid_inline23563 = pool_tid_inline24532;

                // Phase-fence barrier 35: dependency-only dummy task
                CoreTaskArgs params_phase_fence_barrier_35;
                TaskId params_phase_fence_barrier_35_deps[3];
                uint32_t params_phase_fence_barrier_35_deps_count = 0;
                if (topk_tid_inline24347.is_valid()) params_phase_fence_barrier_35_deps[params_phase_fence_barrier_35_deps_count++] = topk_tid_inline24347;
                if (pool_tid_inline24532.is_valid()) params_phase_fence_barrier_35_deps[params_phase_fence_barrier_35_deps_count++] = pool_tid_inline24532;
                if (ret__tmp_v0_guarded_0_34.is_valid()) params_phase_fence_barrier_35_deps[params_phase_fence_barrier_35_deps_count++] = ret__tmp_v0_guarded_0_34;
                params_phase_fence_barrier_35.set_dependencies(params_phase_fence_barrier_35_deps, params_phase_fence_barrier_35_deps_count);
                TaskId selection_ready_inline23564 = TaskId::invalid();
                if (params_phase_fence_barrier_35_deps_count > 0) {
                    TaskOutputTensors phase_fence_barrier_35_outs = rt_submit_dummy_task(params_phase_fence_barrier_35);
                    selection_ready_inline23564 = phase_fence_barrier_35_outs.task_id();
                }
                int64_t t__tmp_v982 = (int64_t)layer_window_cache_inline23453.shapes[0];
                int64_t window_rows_inline24657 = (t__tmp_v982 * 128);
                int64_t t__tmp_v983 = (int64_t)layer_compressed_cache_inline23456.shapes[0];
                int64_t compressed_rows_inline24658 = (t__tmp_v983 * 128);
                uint32_t w_inline24659_shapes[2] = {static_cast<uint32_t>(window_rows_inline24657), 512};
                Tensor w_inline24659 = layer_window_cache_inline23453.reshape(w_inline24659_shapes, 2);
                uint32_t ws_inline24660_shapes[2] = {static_cast<uint32_t>(window_rows_inline24657), 16};
                Tensor ws_inline24660 = layer_window_cache_scale_inline23454.reshape(ws_inline24660_shapes, 2);
                uint32_t c_inline24661_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline24658), 256};
                Tensor c_inline24661 = layer_compressed_cache_inline23456.reshape(c_inline24661_shapes, 2);
                uint32_t cs_inline24662_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline24658), 32};
                Tensor cs_inline24662 = layer_compressed_cache_scale_inline23457.reshape(cs_inline24662_shapes, 2);
                int64_t selected_rows_inline24663 = (LOCAL_T_DYN * 640);
                uint32_t dst_inline24664_shapes[2] = {static_cast<uint32_t>(selected_rows_inline24663), 512};
                Tensor dst_inline24664 = selected_inline23566.reshape(dst_inline24664_shapes, 2);
                int64_t workers_inline24665 = std::max<int64_t>(std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) * 20), 64), 1);

                // Spmd c1a_gather_spmd: c1a_gather
                CoreTaskArgs params_t166;
                params_t166.add_input(ext_window_indices);
                params_t166.add_input(w_inline24659);
                params_t166.add_input(ws_inline24660);
                params_t166.add_input(topk_indices_inline23445__rv_v1);
                params_t166.add_input(c_inline24661);
                params_t166.add_input(cs_inline24662);
                params_t166.add_inout(dst_inline24664);
                params_t166.add_inout(indices_inline23567);
                params_t166.add_scalar(attention_count_inline23434);
                params_t166.add_scalar(workers_inline24665);
                params_t166.launch_spec.set_block_num(workers_inline24665);
                TaskId params_t166_deps[1];
                uint32_t params_t166_deps_count = 0;
                if (selection_ready_inline23564.is_valid()) params_t166_deps[params_t166_deps_count++] = selection_ready_inline23564;
                params_t166.set_dependencies(params_t166_deps, params_t166_deps_count);
                TaskOutputTensors task_166_outs = rt_submit_aiv_task(60, params_t166);
                TaskId gathered_inline24666 = task_166_outs.task_id();
                TaskId gather_tid_inline23571 = gathered_inline24666;
                int64_t head_rows_inline24705 = (LOCAL_T_DYN * 64);
                int64_t cache_rows_inline24706 = (LOCAL_T_DYN * 640);
                uint32_t qflat_inline24707_shapes[2] = {static_cast<uint32_t>(head_rows_inline24705), 512};
                Tensor qflat_inline24707 = query_inline23527__ssa_v1.reshape(qflat_inline24707_shapes, 2);
                uint32_t kflat_inline24708_shapes[2] = {static_cast<uint32_t>(cache_rows_inline24706), 512};
                Tensor kflat_inline24708 = selected_inline23566.reshape(kflat_inline24708_shapes, 2);
                uint32_t oflat_inline24709_shapes[2] = {static_cast<uint32_t>(head_rows_inline24705), 512};
                Tensor oflat_inline24709 = attended_inline23565.reshape(oflat_inline24709_shapes, 2);
                int64_t workers_inline24710 = std::max<int64_t>(std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) * 2), 32), 1);

                // Group sparse_online_cube: MixedKernels (AIC + AIV lanes)
                CoreTaskArgs params_t167;
                params_t167.add_inout(oflat_inline24709);
                params_t167.add_input(qflat_inline24707);
                params_t167.add_input(kflat_inline24708);
                params_t167.add_input(indices_inline23567);
                params_t167.add_input(layer_sink_inline23470);
                params_t167.add_scalar(attention_count_inline23434);
                params_t167.add_scalar(workers_inline24710);
                MixedKernels mixed_167 = {61, 62, INVALID_KERNEL_ID};
                params_t167.launch_spec.set_block_num(workers_inline24710);
                TaskId params_t167_deps[1];
                uint32_t params_t167_deps_count = 0;
                params_t167_deps[params_t167_deps_count++] = gathered_inline24666;
                params_t167.set_dependencies(params_t167_deps, params_t167_deps_count);
                TaskOutputTensors task_167_outs = rt_submit_task(mixed_167, params_t167);
                for (int64_t group_start_inline24755 = 0; group_start_inline24755 < LOCAL_T_DYN; group_start_inline24755 += 80) {
                    uint32_t payload_all_inline24765_ci_shapes[2] = {640, 4096};
                    TensorCreateInfo payload_all_inline24765_ci(payload_all_inline24765_ci_shapes, 2, DataType::FP8E4M3FN);
                    uint32_t scales_all_inline24766_ci_shapes[2] = {8, 10240};
                    TensorCreateInfo scales_all_inline24766_ci(scales_all_inline24766_ci_shapes, 2, DataType::FP8E8M0);
                    uint32_t projected_inline24843_ci_shapes[2] = {80, 8192};
                    TensorCreateInfo projected_inline24843_ci(projected_inline24843_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t payload_inline24846_ci_shapes[2] = {80, 8192};
                    TensorCreateInfo payload_inline24846_ci(payload_inline24846_ci_shapes, 2, DataType::FP8E4M3FN);
                    uint32_t activation_scale_backing_inline24847_ci_shapes[2] = {1, 20480};
                    TensorCreateInfo activation_scale_backing_inline24847_ci(activation_scale_backing_inline24847_ci_shapes, 2, DataType::FP8E8M0);
                    TaskOutputTensors alloc_66 = alloc_tensors(payload_all_inline24765_ci, scales_all_inline24766_ci, projected_inline24843_ci, payload_inline24846_ci, activation_scale_backing_inline24847_ci);
                    const Tensor& payload_all_inline24765 = alloc_66.get_ref(0);
                    const Tensor& scales_all_inline24766 = alloc_66.get_ref(1);
                    const Tensor& projected_inline24843 = alloc_66.get_ref(2);
                    const Tensor& payload_inline24846 = alloc_66.get_ref(3);
                    const Tensor& activation_scale_backing_inline24847 = alloc_66.get_ref(4);
                    int64_t group_rows_inline24756 = std::min<int64_t>((LOCAL_T_DYN - group_start_inline24755), 80);
                    int64_t active_rows_inline24757 = std::max<int64_t>(std::min<int64_t>(group_rows_inline24756, (static_cast<int64_t>(attention_count_inline23434) - group_start_inline24755)), 0);
                    uint32_t attended_group_inline24758_offsets[2] = {static_cast<uint32_t>(group_start_inline24755), 0};
                    uint32_t attended_group_inline24758_shapes[2] = {(attended_group_inline24758_offsets[0] >= attended_inline23565.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline24756), attended_inline23565.shapes[0] - attended_group_inline24758_offsets[0])), (attended_group_inline24758_offsets[1] >= attended_inline23565.shapes[1] ? 0u : std::min<uint32_t>(32768, attended_inline23565.shapes[1] - attended_group_inline24758_offsets[1]))};
                    Tensor attended_group_inline24758 = attended_inline23565.view(attended_group_inline24758_shapes, attended_group_inline24758_offsets);
                    uint32_t cosine_group_inline24759_offsets[2] = {static_cast<uint32_t>(group_start_inline24755), 0};
                    uint32_t cosine_group_inline24759_shapes[2] = {(cosine_group_inline24759_offsets[0] >= compressed_token_rope_cos_inline23437.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline24756), compressed_token_rope_cos_inline23437.shapes[0] - cosine_group_inline24759_offsets[0])), (cosine_group_inline24759_offsets[1] >= compressed_token_rope_cos_inline23437.shapes[1] ? 0u : std::min<uint32_t>(32, compressed_token_rope_cos_inline23437.shapes[1] - cosine_group_inline24759_offsets[1]))};
                    Tensor cosine_group_inline24759 = compressed_token_rope_cos_inline23437.view(cosine_group_inline24759_shapes, cosine_group_inline24759_offsets);
                    uint32_t sine_group_inline24760_offsets[2] = {static_cast<uint32_t>(group_start_inline24755), 0};
                    uint32_t sine_group_inline24760_shapes[2] = {(sine_group_inline24760_offsets[0] >= compressed_token_rope_sin_inline23438.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline24756), compressed_token_rope_sin_inline23438.shapes[0] - sine_group_inline24760_offsets[0])), (sine_group_inline24760_offsets[1] >= compressed_token_rope_sin_inline23438.shapes[1] ? 0u : std::min<uint32_t>(32, compressed_token_rope_sin_inline23438.shapes[1] - sine_group_inline24760_offsets[1]))};
                    Tensor sine_group_inline24760 = compressed_token_rope_sin_inline23438.view(sine_group_inline24760_shapes, sine_group_inline24760_offsets);
                    uint32_t output_group_inline24761_offsets[2] = {static_cast<uint32_t>(group_start_inline24755), 0};
                    uint32_t output_group_inline24761_shapes[2] = {(output_group_inline24761_offsets[0] >= attn_out_inline23572.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline24756), attn_out_inline23572.shapes[0] - output_group_inline24761_offsets[0])), (output_group_inline24761_offsets[1] >= attn_out_inline23572.shapes[1] ? 0u : std::min<uint32_t>(5120, attn_out_inline23572.shapes[1] - output_group_inline24761_offsets[1]))};
                    Tensor output_group_inline24761 = attn_out_inline23572.view(output_group_inline24761_shapes, output_group_inline24761_offsets);
                    int64_t t__tmp_v999 = (int64_t)attended_group_inline24758.shapes[0];
                    int64_t last_input_row_inline24764 = std::max<int64_t>((t__tmp_v999 - 1), 0);
                    int32_t t__tmp_v1000 = rt_available_cluster_count();
                    int64_t quant_workers_inline24769 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v1000) * 2), 64);
                    Tensor ret__tmp_v0_guarded_0_37 = ext_engram_lookup;
                    Tensor ret__tmp_v0_guarded_1_27 = ext_engram_lookup;
                    if ((0 < quant_workers_inline24769)) {

                        // Spmd rope_quant_o_pooled_spmd_2: rope_quant_o_pooled
                        CoreTaskArgs params_t168;
                        params_t168.add_inout(payload_all_inline24765);
                        params_t168.add_inout(scales_all_inline24766);
                        params_t168.add_input(cosine_group_inline24759);
                        params_t168.add_input(sine_group_inline24760);
                        params_t168.add_input(attended_group_inline24758);
                        params_t168.add_scalar(quant_workers_inline24769);
                        params_t168.add_scalar(active_rows_inline24757);
                        params_t168.add_scalar(last_input_row_inline24764);
                        params_t168.launch_spec.set_block_num(quant_workers_inline24769);
                        rt_submit_aiv_task(31, params_t168);
                        ret__tmp_v0_guarded_0_37 = payload_all_inline24765;
                        ret__tmp_v0_guarded_1_27 = scales_all_inline24766;
                    } else {
                        ret__tmp_v0_guarded_0_37 = payload_all_inline24765;
                        ret__tmp_v0_guarded_1_27 = scales_all_inline24766;
                    }
                    Tensor payload_all_inline24765__rv_v2 = ret__tmp_v0_guarded_0_37;
                    Tensor scales_all_inline24766__rv_v2 = ret__tmp_v0_guarded_1_27;
                    int32_t t__tmp_v1003 = rt_available_cluster_count();
                    int64_t oa_workers_inline24844 = std::min<int64_t>(static_cast<int64_t>(t__tmp_v1003), 32);
                    int64_t oa_workers_v1_inline24845 = std::max<int64_t>(oa_workers_inline24844, 1);

                    // Spmd proj_o_a_2: _o_proj_a
                    CoreTaskArgs params_t169;
                    params_t169.add_input(payload_all_inline24765__rv_v2);
                    params_t169.add_input(layer_wo_a_inline23471);
                    params_t169.add_input(scales_all_inline24766__rv_v2);
                    params_t169.add_input(layer_wo_a_scale_inline23472);
                    params_t169.add_inout(projected_inline24843);
                    params_t169.add_scalar((uint64_t)8192);
                    params_t169.launch_spec.set_block_num(oa_workers_v1_inline24845);
                    rt_submit_aic_task(32, params_t169);
                    int32_t t__tmp_v1004 = rt_available_cluster_count();
                    int64_t output_workers_inline24849 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v1004) * 2), 64);
                    Tensor ret__tmp_v0_guarded_0_38 = ext_engram_lookup;
                    Tensor ret__tmp_v0_guarded_1_28 = ext_engram_lookup;
                    if ((0 < output_workers_inline24849)) {

                        // Spmd quant_o_a_spmd_2: quant_o_a
                        CoreTaskArgs params_t170;
                        params_t170.add_inout(activation_scale_backing_inline24847);
                        params_t170.add_inout(payload_inline24846);
                        params_t170.add_input(projected_inline24843);
                        params_t170.add_scalar(output_workers_inline24849);
                        params_t170.add_scalar(active_rows_inline24757);
                        params_t170.launch_spec.set_block_num(output_workers_inline24849);
                        rt_submit_aiv_task(33, params_t170);
                        ret__tmp_v0_guarded_0_38 = activation_scale_backing_inline24847;
                        ret__tmp_v0_guarded_1_28 = payload_inline24846;
                    } else {
                        ret__tmp_v0_guarded_0_38 = activation_scale_backing_inline24847;
                        ret__tmp_v0_guarded_1_28 = payload_inline24846;
                    }
                    Tensor activation_scale_backing_inline24847__rv_v2 = ret__tmp_v0_guarded_0_38;
                    Tensor payload_inline24846__rv_v2 = ret__tmp_v0_guarded_1_28;
                    int64_t capacity_inline24856 = (int64_t)output_group_inline24761.shapes[0];

                    // Spmd proj_o_b_2: _o_proj_b
                    CoreTaskArgs params_t171;
                    params_t171.add_input(payload_inline24846__rv_v2);
                    params_t171.add_input(layer_wo_b_inline23473);
                    params_t171.add_input(activation_scale_backing_inline24847__rv_v2);
                    params_t171.add_input(layer_wo_b_scale_inline23474);
                    params_t171.add_inout(output_group_inline24761);
                    params_t171.add_scalar(capacity_inline24856);
                    params_t171.add_scalar((uint64_t)8192);
                    params_t171.launch_spec.set_block_num(20);
                    rt_submit_aic_task(34, params_t171);
                }
                TaskId hc_post_deps_inline23573[2]; for (int64_t __init_i = 0; __init_i < 2; ++__init_i) hc_post_deps_inline23573[__init_i] = TaskId::invalid();
                uint32_t residual_flat_inline24890_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                Tensor residual_flat_inline24890 = ext_x_hc.reshape(residual_flat_inline24890_shapes, 2);
                uint32_t residual_mix_flat_inline24891_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 16};
                Tensor residual_mix_flat_inline24891 = residual_mix_inline23520.reshape(residual_mix_flat_inline24891_shapes, 2);
                uint32_t output_flat_inline24892_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                Tensor output_flat_inline24892 = attention_hidden_inline23443.reshape(output_flat_inline24892_shapes, 2);
                int64_t total_work_inline24893 = (LOCAL_T_DYN * 2);
                int64_t full_work_inline24894 = ((total_work_inline24893 / 64) * 64);
                int64_t tail_work_inline24895 = ((total_work_inline24893 - full_work_inline24894) * 4);
                int64_t workers_inline24896 = std::min<int64_t>((total_work_inline24893 * 4), 64);
                if ((0 < workers_inline24896)) {

                    // Spmd hc_post_spmd_3: hc_post
                    CoreTaskArgs params_t172;
                    params_t172.add_inout(output_flat_inline24892);
                    params_t172.add_input(attn_out_inline23572);
                    params_t172.add_input(residual_flat_inline24890);
                    params_t172.add_input(post_mix_inline23519__ssa_v1);
                    params_t172.add_input(residual_mix_flat_inline24891);
                    params_t172.add_scalar(full_work_inline24894);
                    params_t172.add_scalar(tail_work_inline24895);
                    params_t172.launch_spec.set_block_num(workers_inline24896);
                    TaskId params_t172_deps[2];
                    uint32_t params_t172_deps_count = 0;
                    if (hc_post_deps_inline23573[0].is_valid()) params_t172_deps[params_t172_deps_count++] = hc_post_deps_inline23573[0];
                    if (hc_post_deps_inline23573[1].is_valid()) params_t172_deps[params_t172_deps_count++] = hc_post_deps_inline23573[1];
                    params_t172.set_dependencies(params_t172_deps, params_t172_deps_count);
                    TaskOutputTensors task_172_outs = rt_submit_aiv_task(35, params_t172);
                } else {

                    // Phase-fence barrier 36: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_36;
                    TaskId params_phase_fence_barrier_36_deps[2];
                    uint32_t params_phase_fence_barrier_36_deps_count = 0;
                    if (hc_post_deps_inline23573[0].is_valid()) params_phase_fence_barrier_36_deps[params_phase_fence_barrier_36_deps_count++] = hc_post_deps_inline23573[0];
                    if (hc_post_deps_inline23573[1].is_valid()) params_phase_fence_barrier_36_deps[params_phase_fence_barrier_36_deps_count++] = hc_post_deps_inline23573[1];
                    params_phase_fence_barrier_36.set_dependencies(params_phase_fence_barrier_36_deps, params_phase_fence_barrier_36_deps_count);
                    TaskId spmd_empty_tid_30 = TaskId::invalid();
                    if (params_phase_fence_barrier_36_deps_count > 0) {
                        TaskOutputTensors phase_fence_barrier_36_outs = rt_submit_dummy_task(params_phase_fence_barrier_36);
                        spmd_empty_tid_30 = phase_fence_barrier_36_outs.task_id();
                    }
                }
                attention_pre_mix_inline23444__rv_v2 = attention_pre_mix_inline23444__ssa_v1;
                topk_indices_inline23445__rv_v4 = topk_indices_inline23445__rv_v1;
                pool_block_ids_inline23448__rv_v2 = pool_block_ids_inline23448__rv_v1;
            } else {
                Tensor topk_indices_inline23445__rv_v3 = ext_engram_lookup;
                Tensor attention_pre_mix_inline23444__rv_v1 = ext_engram_lookup;
                if ((((layer_inline23452 - 20) % 4) == 0)) {
                    uint32_t post_mix_inline24944_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 4};
                    TensorCreateInfo post_mix_inline24944_ci(post_mix_inline24944_ci_shapes, 2, DataType::FLOAT32);
                    uint32_t residual_mix_inline24945_ci_shapes[3] = {static_cast<uint32_t>(LOCAL_T_DYN), 4, 4};
                    TensorCreateInfo residual_mix_inline24945_ci(residual_mix_inline24945_ci_shapes, 3, DataType::FLOAT32);
                    uint32_t x_norm_inline24946_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 5120};
                    TensorCreateInfo x_norm_inline24946_ci(x_norm_inline24946_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t qr_inline24948_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 1280};
                    TensorCreateInfo qr_inline24948_ci(qr_inline24948_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t query_inline24949_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 32768};
                    TensorCreateInfo query_inline24949_ci(query_inline24949_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t kv_inline24950_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 512};
                    TensorCreateInfo kv_inline24950_ci(kv_inline24950_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t kv_projection_inline24951_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 512};
                    TensorCreateInfo kv_projection_inline24951_ci(kv_projection_inline24951_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t projected_inline25276_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 4096};
                    TensorCreateInfo projected_inline25276_ci(projected_inline25276_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t query_inline25281_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 2048};
                    TensorCreateInfo query_inline25281_ci(query_inline25281_ci_shapes, 2, DataType::UINT8);
                    uint32_t query_scale_inline25282_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 128};
                    TensorCreateInfo query_scale_inline25282_ci(query_scale_inline25282_ci_shapes, 2, DataType::FP8E8M0);
                    uint32_t weights_inline25284_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 32};
                    TensorCreateInfo weights_inline25284_ci(weights_inline25284_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t partials_inline25360_ci_shapes[2] = {static_cast<uint32_t>((LOCAL_T_DYN * 4)), 32};
                    TensorCreateInfo partials_inline25360_ci(partials_inline25360_ci_shapes, 2, DataType::FLOAT32);
                    uint32_t prefix_inline25387_ci_shapes[2] = {1, static_cast<uint32_t>((LOCAL_T_DYN + 1))};
                    TensorCreateInfo prefix_inline25387_ci(prefix_inline25387_ci_shapes, 2, DataType::INT32);
                    uint32_t attended_inline24968_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 32768};
                    TensorCreateInfo attended_inline24968_ci(attended_inline24968_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t attn_out_inline24978_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 5120};
                    TensorCreateInfo attn_out_inline24978_ci(attn_out_inline24978_ci_shapes, 2, DataType::BFLOAT16);
                    TaskOutputTensors alloc_67 = alloc_tensors(post_mix_inline24944_ci, residual_mix_inline24945_ci, x_norm_inline24946_ci, qr_inline24948_ci, query_inline24949_ci, kv_inline24950_ci, kv_projection_inline24951_ci, projected_inline25276_ci, query_inline25281_ci, query_scale_inline25282_ci, weights_inline25284_ci, partials_inline25360_ci, prefix_inline25387_ci, attended_inline24968_ci, attn_out_inline24978_ci);
                    const Tensor& post_mix_inline24944 = alloc_67.get_ref(0);
                    const Tensor& residual_mix_inline24945 = alloc_67.get_ref(1);
                    const Tensor& x_norm_inline24946 = alloc_67.get_ref(2);
                    const Tensor& qr_inline24948 = alloc_67.get_ref(3);
                    const Tensor& query_inline24949 = alloc_67.get_ref(4);
                    const Tensor& kv_inline24950 = alloc_67.get_ref(5);
                    const Tensor& kv_projection_inline24951 = alloc_67.get_ref(6);
                    const Tensor& projected_inline25276 = alloc_67.get_ref(7);
                    const Tensor& query_inline25281 = alloc_67.get_ref(8);
                    const Tensor& query_scale_inline25282 = alloc_67.get_ref(9);
                    const Tensor& weights_inline25284 = alloc_67.get_ref(10);
                    const Tensor& partials_inline25360 = alloc_67.get_ref(11);
                    const Tensor& prefix_inline25387 = alloc_67.get_ref(12);
                    const Tensor& attended_inline24968 = alloc_67.get_ref(13);
                    const Tensor& attn_out_inline24978 = alloc_67.get_ref(14);
                    int64_t reindex_index_source_id_inline23484 = (((layer_inline23452 - 24) / 4) + 4);
                    uint32_t reindex_index_cache_inline23485_offsets[4] = {static_cast<uint32_t>((index_blocks_per_source_inline23442 * 3)), 0, 0, 0};
                    uint32_t reindex_index_cache_inline23485_shapes[4] = {(reindex_index_cache_inline23485_offsets[0] >= ext_index_cache_pool.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(index_blocks_per_source_inline23442), ext_index_cache_pool.shapes[0] - reindex_index_cache_inline23485_offsets[0])), (reindex_index_cache_inline23485_offsets[1] >= ext_index_cache_pool.shapes[1] ? 0u : std::min<uint32_t>(128, ext_index_cache_pool.shapes[1] - reindex_index_cache_inline23485_offsets[1])), (reindex_index_cache_inline23485_offsets[2] >= ext_index_cache_pool.shapes[2] ? 0u : std::min<uint32_t>(1, ext_index_cache_pool.shapes[2] - reindex_index_cache_inline23485_offsets[2])), (reindex_index_cache_inline23485_offsets[3] >= ext_index_cache_pool.shapes[3] ? 0u : std::min<uint32_t>(64, ext_index_cache_pool.shapes[3] - reindex_index_cache_inline23485_offsets[3]))};
                    Tensor reindex_index_cache_inline23485 = ext_index_cache_pool.view(reindex_index_cache_inline23485_shapes, reindex_index_cache_inline23485_offsets);
                    uint32_t reindex_index_cache_scale_inline23486_offsets[4] = {static_cast<uint32_t>((index_blocks_per_source_inline23442 * 3)), 0, 0, 0};
                    uint32_t reindex_index_cache_scale_inline23486_shapes[4] = {(reindex_index_cache_scale_inline23486_offsets[0] >= ext_index_cache_scale_pool.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(index_blocks_per_source_inline23442), ext_index_cache_scale_pool.shapes[0] - reindex_index_cache_scale_inline23486_offsets[0])), (reindex_index_cache_scale_inline23486_offsets[1] >= ext_index_cache_scale_pool.shapes[1] ? 0u : std::min<uint32_t>(128, ext_index_cache_scale_pool.shapes[1] - reindex_index_cache_scale_inline23486_offsets[1])), (reindex_index_cache_scale_inline23486_offsets[2] >= ext_index_cache_scale_pool.shapes[2] ? 0u : std::min<uint32_t>(1, ext_index_cache_scale_pool.shapes[2] - reindex_index_cache_scale_inline23486_offsets[2])), (reindex_index_cache_scale_inline23486_offsets[3] >= ext_index_cache_scale_pool.shapes[3] ? 0u : std::min<uint32_t>(4, ext_index_cache_scale_pool.shapes[3] - reindex_index_cache_scale_inline23486_offsets[3]))};
                    Tensor reindex_index_cache_scale_inline23486 = ext_index_cache_scale_pool.view(reindex_index_cache_scale_inline23486_shapes, reindex_index_cache_scale_inline23486_offsets);
                    uint32_t reindex_index_wq_b_scale_inline23487_offsets[2] = {static_cast<uint32_t>(((((layer_inline23452 - 24) / 4) * 40) + 160)), 0};
                    uint32_t reindex_index_wq_b_scale_inline23487_shapes[2] = {(reindex_index_wq_b_scale_inline23487_offsets[0] >= ext_index_wq_b_scale.shapes[0] ? 0u : std::min<uint32_t>(40, ext_index_wq_b_scale.shapes[0] - reindex_index_wq_b_scale_inline23487_offsets[0])), (reindex_index_wq_b_scale_inline23487_offsets[1] >= ext_index_wq_b_scale.shapes[1] ? 0u : std::min<uint32_t>(4096, ext_index_wq_b_scale.shapes[1] - reindex_index_wq_b_scale_inline23487_offsets[1]))};
                    Tensor reindex_index_wq_b_scale_inline23487 = ext_index_wq_b_scale.view(reindex_index_wq_b_scale_inline23487_shapes, reindex_index_wq_b_scale_inline23487_offsets);
                    uint32_t reindex_index_wq_b_inline23488_offsets[3] = {static_cast<uint32_t>(reindex_index_source_id_inline23484), 0, 0};
                    uint32_t reindex_index_wq_b_inline23488_shapes[3] = {(reindex_index_wq_b_inline23488_offsets[0] >= ext_index_wq_b.shapes[0] ? 0u : std::min<uint32_t>(1, ext_index_wq_b.shapes[0] - reindex_index_wq_b_inline23488_offsets[0])), (reindex_index_wq_b_inline23488_offsets[1] >= ext_index_wq_b.shapes[1] ? 0u : std::min<uint32_t>(1280, ext_index_wq_b.shapes[1] - reindex_index_wq_b_inline23488_offsets[1])), (reindex_index_wq_b_inline23488_offsets[2] >= ext_index_wq_b.shapes[2] ? 0u : std::min<uint32_t>(4096, ext_index_wq_b.shapes[2] - reindex_index_wq_b_inline23488_offsets[2]))};
                    Tensor reindex_index_wq_b_inline23488_view = ext_index_wq_b.view(reindex_index_wq_b_inline23488_shapes, reindex_index_wq_b_inline23488_offsets);
                    bool reindex_index_wq_b_inline23488_empty = reindex_index_wq_b_inline23488_view.numel() == 0;
                    Tensor reindex_index_wq_b_inline23488 = reindex_index_wq_b_inline23488_view;
                    reindex_index_wq_b_inline23488.ndims = 2;
                    reindex_index_wq_b_inline23488.shapes[0] = reindex_index_wq_b_inline23488_view.shapes[1];
                    reindex_index_wq_b_inline23488.strides[0] = reindex_index_wq_b_inline23488_view.strides[1];
                    reindex_index_wq_b_inline23488.shapes[1] = reindex_index_wq_b_inline23488_view.shapes[2];
                    reindex_index_wq_b_inline23488.strides[1] = reindex_index_wq_b_inline23488_view.strides[2];
                    if (reindex_index_wq_b_inline23488_empty) reindex_index_wq_b_inline23488.shapes[0] = 0;
                    uint64_t reindex_index_wq_b_inline23488_expected_stride = 1;
                    bool reindex_index_wq_b_inline23488_contiguous = true;
                    for (int32_t i = static_cast<int32_t>(reindex_index_wq_b_inline23488.ndims) - 1; i >= 0; --i) {
                      reindex_index_wq_b_inline23488_contiguous = reindex_index_wq_b_inline23488_contiguous && (reindex_index_wq_b_inline23488.strides[i] == reindex_index_wq_b_inline23488_expected_stride);
                      reindex_index_wq_b_inline23488_expected_stride *= reindex_index_wq_b_inline23488.shapes[i];
                    }
                    reindex_index_wq_b_inline23488.is_contiguous = reindex_index_wq_b_inline23488_empty || reindex_index_wq_b_inline23488_contiguous;
                    uint32_t reindex_index_weights_proj_inline23489_offsets[5] = {static_cast<uint32_t>(reindex_index_source_id_inline23484), 0, 0, 0, 0};
                    uint32_t reindex_index_weights_proj_inline23489_shapes[5] = {(reindex_index_weights_proj_inline23489_offsets[0] >= ext_index_weights_proj.shapes[0] ? 0u : std::min<uint32_t>(1, ext_index_weights_proj.shapes[0] - reindex_index_weights_proj_inline23489_offsets[0])), (reindex_index_weights_proj_inline23489_offsets[1] >= ext_index_weights_proj.shapes[1] ? 0u : std::min<uint32_t>(2, ext_index_weights_proj.shapes[1] - reindex_index_weights_proj_inline23489_offsets[1])), (reindex_index_weights_proj_inline23489_offsets[2] >= ext_index_weights_proj.shapes[2] ? 0u : std::min<uint32_t>(320, ext_index_weights_proj.shapes[2] - reindex_index_weights_proj_inline23489_offsets[2])), (reindex_index_weights_proj_inline23489_offsets[3] >= ext_index_weights_proj.shapes[3] ? 0u : std::min<uint32_t>(16, ext_index_weights_proj.shapes[3] - reindex_index_weights_proj_inline23489_offsets[3])), (reindex_index_weights_proj_inline23489_offsets[4] >= ext_index_weights_proj.shapes[4] ? 0u : std::min<uint32_t>(16, ext_index_weights_proj.shapes[4] - reindex_index_weights_proj_inline23489_offsets[4]))};
                    Tensor reindex_index_weights_proj_inline23489 = ext_index_weights_proj.view(reindex_index_weights_proj_inline23489_shapes, reindex_index_weights_proj_inline23489_offsets);
                    uint32_t x_flat_inline24981_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                    Tensor x_flat_inline24981 = ext_x_hc.reshape(x_flat_inline24981_shapes, 2);
                    int64_t spmd_core_num_18 = ((LOCAL_T_DYN + 7) / 8);
                    Tensor ret__tmp_v0_guarded_0_39 = ext_engram_lookup;
                    TaskId ret__tmp_v0_guarded_1_29 = TaskId::invalid();
                    if ((0 < spmd_core_num_18)) {

                        // Spmd hc_collapse_norm_spmd_3: hc_collapse_norm
                        CoreTaskArgs params_t173;
                        params_t173.add_input(encoder_pre_mix__rv_v1);
                        params_t173.add_input(x_flat_inline24981);
                        params_t173.add_inout(x_norm_inline24946);
                        params_t173.add_input(layer_norm_weight_inline23461);
                        params_t173.add_scalar(LOCAL_T_DYN);
                        params_t173.launch_spec.set_block_num(spmd_core_num_18);
                        TaskOutputTensors task_173_outs = rt_submit_aiv_task(8, params_t173);
                        TaskId ret__tmp_v0_nonempty_1_31 = task_173_outs.task_id();
                        ret__tmp_v0_guarded_0_39 = x_norm_inline24946;
                        ret__tmp_v0_guarded_1_29 = ret__tmp_v0_nonempty_1_31;
                    } else {

                        // Phase-fence barrier 37: dependency-only dummy task
                        CoreTaskArgs params_phase_fence_barrier_37;
                        TaskOutputTensors phase_fence_barrier_37_outs = rt_submit_dummy_task(params_phase_fence_barrier_37);
                        TaskId spmd_empty_tid_31 = phase_fence_barrier_37_outs.task_id();
                        ret__tmp_v0_guarded_0_39 = x_norm_inline24946;
                        ret__tmp_v0_guarded_1_29 = spmd_empty_tid_31;
                    }
                    Tensor x_norm_inline24946__rv_v2 = ret__tmp_v0_guarded_0_39;
                    TaskId ready_inline24982 = ret__tmp_v0_guarded_1_29;
                    TaskId x_ready_inline24947 = ready_inline24982;
                    int64_t padded_rows_inline25029 = (((LOCAL_T_DYN + 79) / 80) * 80);
                    uint32_t payload_inline25030_ci_shapes[2] = {static_cast<uint32_t>(padded_rows_inline25029), 5120};
                    TensorCreateInfo payload_inline25030_ci(payload_inline25030_ci_shapes, 2, DataType::FP8E4M3FN);
                    TaskOutputTensors alloc_68 = alloc_tensors(payload_inline25030_ci);
                    const Tensor& payload_inline25030 = alloc_68.get_ref(0);
                    uint32_t scales_inline25031_ci_shapes[2] = {1, static_cast<uint32_t>((padded_rows_inline25029 * 160))};
                    TensorCreateInfo scales_inline25031_ci(scales_inline25031_ci_shapes, 2, DataType::FP8E8M0);
                    TaskOutputTensors alloc_69 = alloc_tensors(scales_inline25031_ci);
                    const Tensor& scales_inline25031 = alloc_69.get_ref(0);
                    int64_t quant_blocks_inline25032 = ((padded_rows_inline25029 / 16) * 20);
                    int32_t t__tmp_v1041 = rt_available_cluster_count();
                    int64_t workers_inline25033 = std::min<int64_t>(std::min<int64_t>(quant_blocks_inline25032, (static_cast<int64_t>(t__tmp_v1041) * 2)), 32);
                    Tensor ret__tmp_v0_guarded_0_40 = ext_engram_lookup;
                    Tensor ret__tmp_v0_guarded_1_30 = ext_engram_lookup;
                    if ((0 < workers_inline25033)) {

                        // Spmd attention_qkv_input_quant_pooled_spmd_3: attention_qkv_input_quant_pooled
                        CoreTaskArgs params_t174;
                        params_t174.add_inout(payload_inline25030);
                        params_t174.add_inout(scales_inline25031);
                        params_t174.add_input(x_norm_inline24946__rv_v2);
                        params_t174.add_scalar(quant_blocks_inline25032);
                        params_t174.add_scalar(workers_inline25033);
                        params_t174.add_scalar(attention_count_inline23434);
                        params_t174.launch_spec.set_block_num(workers_inline25033);
                        rt_submit_aiv_task(10, params_t174);
                        ret__tmp_v0_guarded_0_40 = payload_inline25030;
                        ret__tmp_v0_guarded_1_30 = scales_inline25031;
                    } else {
                        ret__tmp_v0_guarded_0_40 = payload_inline25030;
                        ret__tmp_v0_guarded_1_30 = scales_inline25031;
                    }
                    Tensor payload_inline25030__rv_v2 = ret__tmp_v0_guarded_0_40;
                    Tensor scales_inline25031__rv_v2 = ret__tmp_v0_guarded_1_30;
                    Tensor proj_payload_inline24953 = payload_inline25030__rv_v2;
                    Tensor proj_scales_inline24954 = scales_inline25031__rv_v2;
                    Tensor query_inline25076 = query_inline24949;
                    int64_t payload_rows_inline25077 = (int64_t)proj_payload_inline24953.shapes[0];
                    uint32_t qb_payload_inline25078_ci_shapes[2] = {static_cast<uint32_t>(payload_rows_inline25077), 1280};
                    TensorCreateInfo qb_payload_inline25078_ci(qb_payload_inline25078_ci_shapes, 2, DataType::FP8E4M3FN);
                    TaskOutputTensors alloc_70 = alloc_tensors(qb_payload_inline25078_ci);
                    const Tensor& qb_payload_inline25078 = alloc_70.get_ref(0);
                    uint32_t qb_scales_inline25079_ci_shapes[2] = {1, static_cast<uint32_t>((payload_rows_inline25077 * 40))};
                    TensorCreateInfo qb_scales_inline25079_ci(qb_scales_inline25079_ci_shapes, 2, DataType::FP8E8M0);
                    TaskOutputTensors alloc_71 = alloc_tensors(qb_scales_inline25079_ci);
                    const Tensor& qb_scales_inline25079 = alloc_71.get_ref(0);
                    Tensor query_inline25076__rv_v2 = query_inline25076;
                    for (int64_t group_start_inline25080 = 0; group_start_inline25080 < attention_count_inline23434; group_start_inline25080 += 80) {
                        uint32_t projected_inline25085_ci_shapes[2] = {80, 1280};
                        TensorCreateInfo projected_inline25085_ci(projected_inline25085_ci_shapes, 2, DataType::BFLOAT16);
                        TaskOutputTensors alloc_72 = alloc_tensors(projected_inline25085_ci);
                        const Tensor& projected_inline25085 = alloc_72.get_ref(0);
                        int64_t group_rows_inline25081 = std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) - group_start_inline25080), 80);
                        uint32_t group_payload_inline25082_offsets[2] = {static_cast<uint32_t>(group_start_inline25080), 0};
                        uint32_t group_payload_inline25082_shapes[2] = {(group_payload_inline25082_offsets[0] >= proj_payload_inline24953.shapes[0] ? 0u : std::min<uint32_t>(80, proj_payload_inline24953.shapes[0] - group_payload_inline25082_offsets[0])), (group_payload_inline25082_offsets[1] >= proj_payload_inline24953.shapes[1] ? 0u : std::min<uint32_t>(5120, proj_payload_inline24953.shapes[1] - group_payload_inline25082_offsets[1]))};
                        Tensor group_payload_inline25082 = proj_payload_inline24953.view(group_payload_inline25082_shapes, group_payload_inline25082_offsets);
                        int64_t scale_start_inline25083 = (group_start_inline25080 * 160);
                        uint32_t group_scales_inline25084_offsets[2] = {0, static_cast<uint32_t>(scale_start_inline25083)};
                        uint32_t group_scales_inline25084_shapes[2] = {(group_scales_inline25084_offsets[0] >= proj_scales_inline24954.shapes[0] ? 0u : std::min<uint32_t>(1, proj_scales_inline24954.shapes[0] - group_scales_inline25084_offsets[0])), (group_scales_inline25084_offsets[1] >= proj_scales_inline24954.shapes[1] ? 0u : std::min<uint32_t>(12800, proj_scales_inline24954.shapes[1] - group_scales_inline25084_offsets[1]))};
                        Tensor group_scales_inline25084 = proj_scales_inline24954.view(group_scales_inline25084_shapes, group_scales_inline25084_offsets);

                        // Spmd proj_q_a_3: _proj_q_a
                        CoreTaskArgs params_t175;
                        params_t175.add_input(group_payload_inline25082);
                        params_t175.add_input(layer_wq_a_inline23462);
                        params_t175.add_input(group_scales_inline25084);
                        params_t175.add_input(layer_wq_a_scale_inline23463);
                        params_t175.add_inout(projected_inline25085);
                        params_t175.add_scalar(group_rows_inline25081);
                        params_t175.launch_spec.set_block_num(20);
                        rt_submit_aic_task(11, params_t175);
                        uint32_t qb_payload_group_inline25086_offsets[2] = {static_cast<uint32_t>(group_start_inline25080), 0};
                        uint32_t qb_payload_group_inline25086_shapes[2] = {(qb_payload_group_inline25086_offsets[0] >= qb_payload_inline25078.shapes[0] ? 0u : std::min<uint32_t>(80, qb_payload_inline25078.shapes[0] - qb_payload_group_inline25086_offsets[0])), (qb_payload_group_inline25086_offsets[1] >= qb_payload_inline25078.shapes[1] ? 0u : std::min<uint32_t>(1280, qb_payload_inline25078.shapes[1] - qb_payload_group_inline25086_offsets[1]))};
                        Tensor qb_payload_group_inline25086 = qb_payload_inline25078.view(qb_payload_group_inline25086_shapes, qb_payload_group_inline25086_offsets);
                        int64_t qb_scale_start_inline25087 = (group_start_inline25080 * 40);
                        uint32_t qb_scales_group_inline25088_offsets[2] = {0, static_cast<uint32_t>(qb_scale_start_inline25087)};
                        uint32_t qb_scales_group_inline25088_shapes[2] = {(qb_scales_group_inline25088_offsets[0] >= qb_scales_inline25079.shapes[0] ? 0u : std::min<uint32_t>(1, qb_scales_inline25079.shapes[0] - qb_scales_group_inline25088_offsets[0])), (qb_scales_group_inline25088_offsets[1] >= qb_scales_inline25079.shapes[1] ? 0u : std::min<uint32_t>(3200, qb_scales_inline25079.shapes[1] - qb_scales_group_inline25088_offsets[1]))};
                        Tensor qb_scales_group_inline25088 = qb_scales_inline25079.view(qb_scales_group_inline25088_shapes, qb_scales_group_inline25088_offsets);
                        for (int64_t q_mt_inline25089 = 0; q_mt_inline25089 < 5; q_mt_inline25089 += 1) {
                            int64_t q_local_inline25090 = (q_mt_inline25089 * 16);
                            int64_t q_start_inline25091 = (group_start_inline25080 + std::min<int64_t>(q_local_inline25090, group_rows_inline25081));
                            int64_t q_rows_inline25092 = std::max<int64_t>(std::min<int64_t>((group_rows_inline25081 - q_local_inline25090), 16), 0);
                            uint32_t projected_tile_inline25093_offsets[2] = {static_cast<uint32_t>(q_local_inline25090), 0};
                            uint32_t projected_tile_inline25093_shapes[2] = {(projected_tile_inline25093_offsets[0] >= projected_inline25085.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(q_rows_inline25092), projected_inline25085.shapes[0] - projected_tile_inline25093_offsets[0])), (projected_tile_inline25093_offsets[1] >= projected_inline25085.shapes[1] ? 0u : std::min<uint32_t>(1280, projected_inline25085.shapes[1] - projected_tile_inline25093_offsets[1]))};
                            Tensor projected_tile_inline25093 = projected_inline25085.view(projected_tile_inline25093_shapes, projected_tile_inline25093_offsets);
                            uint32_t latent_tile_inline25094_offsets[2] = {static_cast<uint32_t>(q_start_inline25091), 0};
                            uint32_t latent_tile_inline25094_shapes[2] = {(latent_tile_inline25094_offsets[0] >= qr_inline24948.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(q_rows_inline25092), qr_inline24948.shapes[0] - latent_tile_inline25094_offsets[0])), (latent_tile_inline25094_offsets[1] >= qr_inline24948.shapes[1] ? 0u : std::min<uint32_t>(1280, qr_inline24948.shapes[1] - latent_tile_inline25094_offsets[1]))};
                            Tensor latent_tile_inline25094 = qr_inline24948.view(latent_tile_inline25094_shapes, latent_tile_inline25094_offsets);
                            uint32_t qb_payload_tile_inline25095_offsets[2] = {static_cast<uint32_t>(q_local_inline25090), 0};
                            uint32_t qb_payload_tile_inline25095_shapes[2] = {(qb_payload_tile_inline25095_offsets[0] >= qb_payload_group_inline25086.shapes[0] ? 0u : std::min<uint32_t>(16, qb_payload_group_inline25086.shapes[0] - qb_payload_tile_inline25095_offsets[0])), (qb_payload_tile_inline25095_offsets[1] >= qb_payload_group_inline25086.shapes[1] ? 0u : std::min<uint32_t>(1280, qb_payload_group_inline25086.shapes[1] - qb_payload_tile_inline25095_offsets[1]))};
                            Tensor qb_payload_tile_inline25095 = qb_payload_group_inline25086.view(qb_payload_tile_inline25095_shapes, qb_payload_tile_inline25095_offsets);
                            int64_t qb_scale_offset_inline25096 = (q_local_inline25090 * 40);
                            uint32_t qb_scales_tile_inline25097_offsets[2] = {0, static_cast<uint32_t>(qb_scale_offset_inline25096)};
                            uint32_t qb_scales_tile_inline25097_shapes[2] = {(qb_scales_tile_inline25097_offsets[0] >= qb_scales_group_inline25088.shapes[0] ? 0u : std::min<uint32_t>(1, qb_scales_group_inline25088.shapes[0] - qb_scales_tile_inline25097_offsets[0])), (qb_scales_tile_inline25097_offsets[1] >= qb_scales_group_inline25088.shapes[1] ? 0u : std::min<uint32_t>(640, qb_scales_group_inline25088.shapes[1] - qb_scales_tile_inline25097_offsets[1]))};
                            Tensor qb_scales_tile_inline25097 = qb_scales_group_inline25088.view(qb_scales_tile_inline25097_shapes, qb_scales_tile_inline25097_offsets);
                            uint32_t weight_row_inline25102_shapes[2] = {1, 1280};
                            Tensor weight_row_inline25102 = layer_q_norm_inline23464.reshape(weight_row_inline25102_shapes, 2);

                            // Spmd attention_q_norm_quant_spmd_3: attention_q_norm_quant_3
                            CoreTaskArgs params_t176;
                            params_t176.add_input(projected_tile_inline25093);
                            params_t176.add_input(weight_row_inline25102);
                            params_t176.add_inout(latent_tile_inline25094);
                            params_t176.add_inout(qb_payload_tile_inline25095);
                            params_t176.add_inout(qb_scales_tile_inline25097);
                            params_t176.add_scalar(q_rows_inline25092);
                            params_t176.launch_spec.set_block_num(5);
                            rt_submit_aiv_task(109, params_t176);
                        }
                        int32_t group_tokens_inline25098 = static_cast<int32_t>(group_rows_inline25081);
                        int32_t t__tmp_v1050 = rt_available_cluster_count();
                        int64_t qb_workers_v1_inline25100 = std::min<int64_t>(static_cast<int64_t>(t__tmp_v1050), 32);
                        int64_t qb_workers_v2_inline25101 = std::max<int64_t>(qb_workers_v1_inline25100, 1);

                        // Spmd proj_q_b_3: _proj_q_b
                        CoreTaskArgs params_t177;
                        params_t177.add_input(qb_payload_group_inline25086);
                        params_t177.add_input(layer_wq_b_inline23465);
                        params_t177.add_input(qb_scales_group_inline25088);
                        params_t177.add_input(layer_wq_b_scale_inline23466);
                        params_t177.add_inout(query_inline25076__rv_v2);
                        params_t177.add_scalar(group_rows_inline25081);
                        params_t177.add_scalar((uint64_t)32768);
                        params_t177.add_scalar(group_start_inline25080);
                        params_t177.launch_spec.set_block_num(qb_workers_v2_inline25101);
                        rt_submit_aic_task(13, params_t177);
                        int32_t t__tmp_v1051 = rt_available_cluster_count();
                        int64_t rope_workers_inline25162 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v1051) * 2), 32);
                        Tensor query_inline25076__rv_v5 = ext_engram_lookup;
                        if ((0 < rope_workers_inline25162)) {

                            // Spmd attention_q_rope_spmd_3: attention_q_rope
                            CoreTaskArgs params_t178;
                            params_t178.add_inout(query_inline25076__rv_v2);
                            params_t178.add_input(compressed_token_rope_cos_inline23437);
                            params_t178.add_input(compressed_token_rope_sin_inline23438);
                            params_t178.add_scalar(rope_workers_inline25162);
                            params_t178.add_scalar(group_start_inline25080);
                            params_t178.add_scalar(group_tokens_inline25098);
                            params_t178.launch_spec.set_block_num(rope_workers_inline25162);
                            rt_submit_aiv_task(14, params_t178);
                            query_inline25076__rv_v5 = query_inline25076__rv_v2;
                        } else {
                            query_inline25076__rv_v5 = query_inline25076__rv_v2;
                        }
                        query_inline25076__rv_v2 = query_inline25076__rv_v5;
                    }
                    Tensor query_inline24949__ssa_v1 = query_inline25076__rv_v2;
                    Tensor qb_payload_inline24956 = qb_payload_inline25078;
                    Tensor qb_scales_inline24957 = qb_scales_inline25079;
                    Tensor kv_projection_inline25196 = kv_projection_inline24951;
                    Tensor kv_inline24950__rv_v2 = kv_inline24950;
                    for (int64_t group_start_inline25197 = 0; group_start_inline25197 < attention_count_inline23434; group_start_inline25197 += 80) {
                        int64_t group_rows_inline25198 = std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) - group_start_inline25197), 80);
                        uint32_t group_payload_inline25199_offsets[2] = {static_cast<uint32_t>(group_start_inline25197), 0};
                        uint32_t group_payload_inline25199_shapes[2] = {(group_payload_inline25199_offsets[0] >= proj_payload_inline24953.shapes[0] ? 0u : std::min<uint32_t>(80, proj_payload_inline24953.shapes[0] - group_payload_inline25199_offsets[0])), (group_payload_inline25199_offsets[1] >= proj_payload_inline24953.shapes[1] ? 0u : std::min<uint32_t>(5120, proj_payload_inline24953.shapes[1] - group_payload_inline25199_offsets[1]))};
                        Tensor group_payload_inline25199 = proj_payload_inline24953.view(group_payload_inline25199_shapes, group_payload_inline25199_offsets);
                        int64_t scale_start_inline25200 = (group_start_inline25197 * 160);
                        uint32_t group_scales_inline25201_offsets[2] = {0, static_cast<uint32_t>(scale_start_inline25200)};
                        uint32_t group_scales_inline25201_shapes[2] = {(group_scales_inline25201_offsets[0] >= proj_scales_inline24954.shapes[0] ? 0u : std::min<uint32_t>(1, proj_scales_inline24954.shapes[0] - group_scales_inline25201_offsets[0])), (group_scales_inline25201_offsets[1] >= proj_scales_inline24954.shapes[1] ? 0u : std::min<uint32_t>(12800, proj_scales_inline24954.shapes[1] - group_scales_inline25201_offsets[1]))};
                        Tensor group_scales_inline25201 = proj_scales_inline24954.view(group_scales_inline25201_shapes, group_scales_inline25201_offsets);

                        // Spmd proj_kv_3: _proj_kv
                        CoreTaskArgs params_t179;
                        params_t179.add_input(group_payload_inline25199);
                        params_t179.add_input(layer_wkv_inline23467);
                        params_t179.add_input(group_scales_inline25201);
                        params_t179.add_input(layer_wkv_scale_inline23468);
                        params_t179.add_inout(kv_projection_inline25196);
                        params_t179.add_scalar(group_rows_inline25198);
                        params_t179.add_scalar(group_start_inline25197);
                        params_t179.launch_spec.set_block_num(8);
                        params_t179.launch_spec.set_require_sync_start(true);
                        rt_submit_aic_task(15, params_t179);
                        uint32_t projection_group_inline25202_offsets[2] = {static_cast<uint32_t>(group_start_inline25197), 0};
                        uint32_t projection_group_inline25202_shapes[2] = {(projection_group_inline25202_offsets[0] >= kv_projection_inline25196.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline25198), kv_projection_inline25196.shapes[0] - projection_group_inline25202_offsets[0])), (projection_group_inline25202_offsets[1] >= kv_projection_inline25196.shapes[1] ? 0u : std::min<uint32_t>(512, kv_projection_inline25196.shapes[1] - projection_group_inline25202_offsets[1]))};
                        Tensor projection_group_inline25202 = kv_projection_inline25196.view(projection_group_inline25202_shapes, projection_group_inline25202_offsets);
                        uint32_t normalized_group_inline25203_offsets[2] = {static_cast<uint32_t>(group_start_inline25197), 0};
                        uint32_t normalized_group_inline25203_shapes[2] = {(normalized_group_inline25203_offsets[0] >= kv_inline24950__rv_v2.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline25198), kv_inline24950__rv_v2.shapes[0] - normalized_group_inline25203_offsets[0])), (normalized_group_inline25203_offsets[1] >= kv_inline24950__rv_v2.shapes[1] ? 0u : std::min<uint32_t>(512, kv_inline24950__rv_v2.shapes[1] - normalized_group_inline25203_offsets[1]))};
                        Tensor normalized_group_inline25203 = kv_inline24950__rv_v2.view(normalized_group_inline25203_shapes, normalized_group_inline25203_offsets);
                        int32_t group_tokens_inline25204 = static_cast<int32_t>(group_rows_inline25198);
                        int64_t spmd_core_num_19 = ((static_cast<int64_t>(group_tokens_inline25204) + 7) / 8);
                        if ((0 < spmd_core_num_19)) {

                            // Spmd norm_kv_spmd_3: norm_kv
                            CoreTaskArgs params_t180;
                            params_t180.add_input(projection_group_inline25202);
                            params_t180.add_input(layer_kv_norm_inline23469);
                            params_t180.add_inout(normalized_group_inline25203);
                            params_t180.add_scalar(group_tokens_inline25204);
                            params_t180.launch_spec.set_block_num(spmd_core_num_19);
                            rt_submit_aiv_task(16, params_t180);
                        } else {
                        }
                        int32_t t__tmp_v1062 = rt_available_cluster_count();
                        int64_t rope_workers_inline25215 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v1062) * 2), 5);
                        Tensor kv_inline24950__rv_v4 = ext_engram_lookup;
                        if ((0 < rope_workers_inline25215)) {

                            // Spmd rope_kv_spmd_3: rope_kv
                            CoreTaskArgs params_t181;
                            params_t181.add_inout(kv_inline24950__rv_v2);
                            params_t181.add_input(compressed_token_rope_cos_inline23437);
                            params_t181.add_input(compressed_token_rope_sin_inline23438);
                            params_t181.add_scalar(rope_workers_inline25215);
                            params_t181.add_scalar(group_start_inline25197);
                            params_t181.add_scalar(group_tokens_inline25204);
                            params_t181.launch_spec.set_block_num(rope_workers_inline25215);
                            rt_submit_aiv_task(17, params_t181);
                            kv_inline24950__rv_v4 = kv_inline24950__rv_v2;
                        } else {
                            kv_inline24950__rv_v4 = kv_inline24950__rv_v2;
                        }
                        kv_inline24950__rv_v2 = kv_inline24950__rv_v4;
                    }
                    int64_t workers_inline25249 = std::min<int64_t>(static_cast<int64_t>(attention_count_inline23434), 12);
                    int64_t step_inline25250 = std::max<int64_t>(workers_inline25249, 1);
                    if ((0 < workers_inline25249)) {

                        // Spmd pub_win_spmd_3: pub_win_3
                        CoreTaskArgs params_t182;
                        params_t182.add_inout(layer_window_cache_inline23453);
                        params_t182.add_inout(layer_window_cache_scale_inline23454);
                        params_t182.add_input(ext_window_slots);
                        params_t182.add_input(kv_inline24950__rv_v2);
                        params_t182.add_scalar(attention_count_inline23434);
                        params_t182.add_scalar(step_inline25250);
                        params_t182.launch_spec.set_block_num(workers_inline25249);
                        rt_submit_aiv_task(110, params_t182);
                    } else {
                    }
                    int64_t t__tmp_v1065 = (int64_t)pool_block_ids_inline23448__rv_v3.shapes[1];
                    int64_t score_width_inline24958 = (t__tmp_v1065 * 8);
                    for (int64_t group_start_inline25277 = 0; group_start_inline25277 < attention_count_inline23434; group_start_inline25277 += 80) {
                        int64_t rows_inline25278 = std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) - group_start_inline25277), 80);
                        int32_t t__tmp_v1066 = rt_available_cluster_count();
                        int64_t query_workers_inline25279 = std::min<int64_t>(static_cast<int64_t>(t__tmp_v1066), 16);
                        int64_t query_workers_v1_inline25280 = std::max<int64_t>(query_workers_inline25279, 1);

                        // Spmd proj_idx_q_1: _proj_idx_q
                        CoreTaskArgs params_t183;
                        params_t183.add_input(qb_payload_inline24956);
                        params_t183.add_input(reindex_index_wq_b_inline23488);
                        params_t183.add_input(qb_scales_inline24957);
                        params_t183.add_input(reindex_index_wq_b_scale_inline23487);
                        params_t183.add_inout(projected_inline25276);
                        params_t183.add_scalar(rows_inline25278);
                        params_t183.add_scalar((uint64_t)4096);
                        params_t183.add_scalar(group_start_inline25277);
                        params_t183.add_scalar(group_start_inline25277);
                        params_t183.launch_spec.set_block_num(query_workers_v1_inline25280);
                        rt_submit_aic_task(45, params_t183);
                    }
                    TaskId entry_deps_inline25287[1]; for (int64_t __init_i = 0; __init_i < 1; ++__init_i) entry_deps_inline25287[__init_i] = TaskId::invalid();
                    entry_deps_inline25287[0] = x_ready_inline24947;
                    TaskId _submit_deps_buf_inline25288[1]; for (int64_t __init_i = 0; __init_i < 1; ++__init_i) _submit_deps_buf_inline25288[__init_i] = TaskId::invalid();
                    TaskId t__tmp_v1067 = entry_deps_inline25287[0];
                    _submit_deps_buf_inline25288[0] = t__tmp_v1067;

                    // Spmd idx_q_fp4_spmd_1: idx_q_fp4
                    CoreTaskArgs params_t184;
                    params_t184.add_input(projected_inline25276);
                    params_t184.add_input(compressed_token_rope_cos_inline23437);
                    params_t184.add_input(compressed_token_rope_sin_inline23438);
                    params_t184.add_inout(query_inline25281);
                    params_t184.add_inout(query_scale_inline25282);
                    params_t184.add_scalar(attention_count_inline23434);
                    params_t184.launch_spec.set_block_num(32);
                    TaskId params_t184_deps[1];
                    uint32_t params_t184_deps_count = 0;
                    if (_submit_deps_buf_inline25288[0].is_valid()) params_t184_deps[params_t184_deps_count++] = _submit_deps_buf_inline25288[0];
                    params_t184.set_dependencies(params_t184_deps, params_t184_deps_count);
                    TaskOutputTensors task_184_outs = rt_submit_aiv_task(46, params_t184);
                    TaskId quant_tid_inline25290 = task_184_outs.task_id();
                    TaskId query_tid_inline25283 = quant_tid_inline25290;
                    int64_t blocks_inline25361 = ((static_cast<int64_t>(attention_count_inline23434) + 79) / 80);

                    // Spmd proj_idx_w_spmd_1: proj_idx_w
                    CoreTaskArgs params_t185;
                    params_t185.add_inout(partials_inline25360);
                    params_t185.add_input(x_norm_inline24946__rv_v2);
                    params_t185.add_input(reindex_index_weights_proj_inline23489);
                    params_t185.add_scalar(LOCAL_T_DYN);
                    params_t185.add_scalar(blocks_inline25361);
                    params_t185.add_scalar(attention_count_inline23434);
                    params_t185.launch_spec.set_block_num(4);
                    TaskId params_t185_deps[1];
                    uint32_t params_t185_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_29.is_valid()) params_t185_deps[params_t185_deps_count++] = ret__tmp_v0_guarded_1_29;
                    params_t185.set_dependencies(params_t185_deps, params_t185_deps_count);
                    TaskOutputTensors task_185_outs = rt_submit_aic_task(47, params_t185);
                    TaskId cube_tid_inline25362 = task_185_outs.task_id();

                    // Spmd reduce_idx_w_spmd_1: reduce_idx_w
                    CoreTaskArgs params_t186;
                    params_t186.add_inout(weights_inline25284);
                    params_t186.add_input(partials_inline25360);
                    params_t186.add_scalar(LOCAL_T_DYN);
                    params_t186.add_scalar(attention_count_inline23434);
                    params_t186.launch_spec.set_block_num(std::max<int64_t>(std::min<int64_t>(((static_cast<int64_t>(attention_count_inline23434) + 15) / 16), 8), 1));
                    TaskId params_t186_deps[1];
                    uint32_t params_t186_deps_count = 0;
                    params_t186_deps[params_t186_deps_count++] = cube_tid_inline25362;
                    params_t186.set_dependencies(params_t186_deps, params_t186_deps_count);
                    TaskOutputTensors task_186_outs = rt_submit_aiv_task(48, params_t186);
                    TaskId completed_inline25376 = task_186_outs.task_id();
                    TaskId weights_tid_inline25285 = completed_inline25376;
                    uint32_t scores_inline25286_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), static_cast<uint32_t>(score_width_inline24958)};
                    TensorCreateInfo scores_inline25286_ci(scores_inline25286_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_73 = alloc_tensors(scores_inline25286_ci);
                    const Tensor& scores_inline25286 = alloc_73.get_ref(0);
                    Tensor index_query_inline24960 = query_inline25281;
                    Tensor index_query_scale_inline24961 = query_scale_inline25282;
                    Tensor index_weights_inline24962 = weights_inline25284;
                    Tensor scores_inline24963 = scores_inline25286;
                    TaskId query_tid_inline24964 = query_tid_inline25283;
                    TaskId weights_tid_inline24965 = weights_tid_inline25285;
                    int64_t capacity_inline25385 = (int64_t)pool_block_ids_inline23448__rv_v3.shapes[1];
                    int64_t width_inline25386 = (int64_t)scores_inline24963.shapes[1];

                    // Task 187: idx_plan_pool
                    CoreTaskArgs params_t187;
                    params_t187.add_inout(prefix_inline25387);
                    params_t187.add_input(pool_block_counts_inline23449);
                    params_t187.add_scalar(attention_count_inline23434);
                    params_t187.add_scalar(capacity_inline25385);
                    TaskId params_t187_deps[1];
                    uint32_t params_t187_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_29.is_valid()) params_t187_deps[params_t187_deps_count++] = ret__tmp_v0_guarded_1_29;
                    params_t187.set_dependencies(params_t187_deps, params_t187_deps_count);
                    TaskOutputTensors task_187_outs = rt_submit_aiv_task(111, params_t187);
                    TaskId plan_tid_inline25388 = task_187_outs.task_id();
                    int64_t output_panels_inline25392 = ((width_inline25386 + 127) / 128);

                    // Spmd idx_clear_pool_spmd: idx_clear_pool
                    CoreTaskArgs params_t188;
                    params_t188.add_inout(scores_inline24963);
                    params_t188.add_scalar(attention_count_inline23434);
                    params_t188.add_scalar(output_panels_inline25392);
                    params_t188.add_scalar(width_inline25386);
                    params_t188.launch_spec.set_block_num(32);
                    TaskId params_t188_deps[1];
                    uint32_t params_t188_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_29.is_valid()) params_t188_deps[params_t188_deps_count++] = ret__tmp_v0_guarded_1_29;
                    params_t188.set_dependencies(params_t188_deps, params_t188_deps_count);
                    TaskOutputTensors task_188_outs = rt_submit_aiv_task(112, params_t188);
                    TaskId clear_tid_inline25393 = task_188_outs.task_id();

                    // Phase-fence barrier 38: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_38;
                    TaskId params_phase_fence_barrier_38_deps[5];
                    uint32_t params_phase_fence_barrier_38_deps_count = 0;
                    if (ret__tmp_v0_guarded_1_29.is_valid()) params_phase_fence_barrier_38_deps[params_phase_fence_barrier_38_deps_count++] = ret__tmp_v0_guarded_1_29;
                    params_phase_fence_barrier_38_deps[params_phase_fence_barrier_38_deps_count++] = quant_tid_inline25290;
                    params_phase_fence_barrier_38_deps[params_phase_fence_barrier_38_deps_count++] = completed_inline25376;
                    params_phase_fence_barrier_38_deps[params_phase_fence_barrier_38_deps_count++] = plan_tid_inline25388;
                    params_phase_fence_barrier_38_deps[params_phase_fence_barrier_38_deps_count++] = clear_tid_inline25393;
                    params_phase_fence_barrier_38.set_dependencies(params_phase_fence_barrier_38_deps, params_phase_fence_barrier_38_deps_count);
                    TaskId ready_inline25398 = TaskId::invalid();
                    if (params_phase_fence_barrier_38_deps_count > 0) {
                        TaskOutputTensors phase_fence_barrier_38_outs = rt_submit_dummy_task(params_phase_fence_barrier_38);
                        ready_inline25398 = phase_fence_barrier_38_outs.task_id();
                    }
                    int64_t capacity_inline25400 = (int64_t)pool_block_ids_inline23448__rv_v3.shapes[1];
                    int64_t cache_pages_inline25401 = (int64_t)reindex_index_cache_inline23485.shapes[0];
                    int64_t cache_rows_inline25402 = (cache_pages_inline25401 * 128);
                    uint32_t cache_flat_inline25403_shapes[2] = {static_cast<uint32_t>(cache_rows_inline25402), 64};
                    Tensor cache_flat_inline25403 = reindex_index_cache_inline23485.reshape(cache_flat_inline25403_shapes, 2);
                    uint32_t scale_flat_inline25404_shapes[2] = {static_cast<uint32_t>((cache_rows_inline25402 / 8)), 32};
                    Tensor scale_flat_inline25404 = reindex_index_cache_scale_inline23486.reshape(scale_flat_inline25404_shapes, 2);
                    int64_t gathered_rows_inline25406 = (((capacity_inline25400 + 15) / 16) * 128);
                    int64_t total_rows_inline25407 = (LOCAL_T_DYN * gathered_rows_inline25406);
                    int64_t total_scales_inline25408 = (total_rows_inline25407 * 4);
                    uint32_t payloads_inline25409_ci_shapes[2] = {static_cast<uint32_t>(total_rows_inline25407), 64};
                    TensorCreateInfo payloads_inline25409_ci(payloads_inline25409_ci_shapes, 2, DataType::UINT8);
                    TaskOutputTensors alloc_74 = alloc_tensors(payloads_inline25409_ci);
                    const Tensor& payloads_inline25409 = alloc_74.get_ref(0);
                    uint32_t scales_inline25410_ci_shapes[2] = {1, static_cast<uint32_t>(total_scales_inline25408)};
                    TensorCreateInfo scales_inline25410_ci(scales_inline25410_ci_shapes, 2, DataType::UINT8);
                    TaskOutputTensors alloc_75 = alloc_tensors(scales_inline25410_ci);
                    const Tensor& scales_inline25410 = alloc_75.get_ref(0);
                    uint32_t valid_inline25411_ci_shapes[2] = {1, static_cast<uint32_t>(total_rows_inline25407)};
                    TensorCreateInfo valid_inline25411_ci(valid_inline25411_ci_shapes, 2, DataType::INT32);
                    TaskOutputTensors alloc_76 = alloc_tensors(valid_inline25411_ci);
                    const Tensor& valid_inline25411 = alloc_76.get_ref(0);

                    // Spmd idx_gather_pool_spmd: idx_gather_pool
                    CoreTaskArgs params_t189;
                    params_t189.add_input(prefix_inline25387);
                    params_t189.add_input(pool_block_counts_inline23449);
                    params_t189.add_input(ext_token_to_req_indices);
                    params_t189.add_input(c1a_compressed_lens_inline23439);
                    params_t189.add_input(pool_block_ids_inline23448__rv_v3);
                    params_t189.add_input(ext_c1a_index_block_table);
                    params_t189.add_input(cache_flat_inline25403);
                    params_t189.add_input(scale_flat_inline25404);
                    params_t189.add_inout(payloads_inline25409);
                    params_t189.add_inout(scales_inline25410);
                    params_t189.add_inout(valid_inline25411);
                    params_t189.add_scalar(B_DYN);
                    params_t189.add_scalar(FWD_C1A_TABLE_DYN);
                    params_t189.add_scalar(attention_count_inline23434);
                    params_t189.add_scalar(capacity_inline25400);
                    params_t189.add_scalar(cache_pages_inline25401);
                    params_t189.add_scalar(gathered_rows_inline25406);
                    params_t189.launch_spec.set_block_num(32);
                    TaskId params_t189_deps[1];
                    uint32_t params_t189_deps_count = 0;
                    if (ready_inline25398.is_valid()) params_t189_deps[params_t189_deps_count++] = ready_inline25398;
                    params_t189.set_dependencies(params_t189_deps, params_t189_deps_count);
                    TaskOutputTensors task_189_outs = rt_submit_aiv_task(113, params_t189);
                    TaskId gather_tid_inline25414 = task_189_outs.task_id();
                    int32_t t__tmp_v1101 = rt_available_cluster_count();
                    int64_t launch_workers_inline25448 = std::max<int64_t>(std::min<int64_t>(static_cast<int64_t>(t__tmp_v1101), 32), 1);
                    int64_t width_inline25449 = (int64_t)scores_inline24963.shapes[1];

                    // Group _pool_index_scores: MixedKernels (AIC + AIV lanes)
                    CoreTaskArgs params_t190;
                    params_t190.add_input(index_query_inline24960);
                    params_t190.add_input(index_query_scale_inline24961);
                    params_t190.add_input(index_weights_inline24962);
                    params_t190.add_input(payloads_inline25409);
                    params_t190.add_input(scales_inline25410);
                    params_t190.add_input(ext_c1a_index_block_table);
                    params_t190.add_input(ext_token_to_req_indices);
                    params_t190.add_input(c1a_compressed_lens_inline23439);
                    params_t190.add_inout(scores_inline24963);
                    params_t190.add_input(pool_block_counts_inline23449);
                    params_t190.add_input(valid_inline25411);
                    params_t190.add_scalar(width_inline25449);
                    params_t190.add_scalar((uint64_t)0);
                    params_t190.add_scalar(FWD_C1A_TABLE_DYN);
                    params_t190.add_scalar((uint64_t)static_cast<int32_t>(0));
                    params_t190.add_scalar(gathered_rows_inline25406);
                    params_t190.add_scalar(attention_count_inline23434);
                    MixedKernels mixed_190 = {114, 115, INVALID_KERNEL_ID};
                    params_t190.launch_spec.set_block_num(launch_workers_inline25448);
                    params_t190.launch_spec.set_require_sync_start(true);
                    TaskId params_t190_deps[1];
                    uint32_t params_t190_deps_count = 0;
                    params_t190_deps[params_t190_deps_count++] = gather_tid_inline25414;
                    params_t190.set_dependencies(params_t190_deps, params_t190_deps_count);
                    TaskOutputTensors task_190_outs = rt_submit_task(mixed_190, params_t190);
                    TaskId score_tid_inline25450 = task_190_outs.task_id();
                    TaskId result_inline25399 = score_tid_inline25450;
                    TaskId scores_tid_inline24966 = result_inline25399;
                    int64_t width_inline25451 = (int64_t)scores_inline24963.shapes[1];
                    TaskId result_inline25453 = TaskId::invalid();
                    Tensor topk_indices_inline23445__rv_v2 = ext_engram_lookup;
                    if ((width_inline25451 <= 512)) {

                        // Spmd pool_topk_short_spmd: pool_topk_short
                        CoreTaskArgs params_t191;
                        params_t191.add_input(ext_token_to_req_indices);
                        params_t191.add_inout(topk_indices_inline23445__rv_v5);
                        params_t191.add_input(pool_block_ids_inline23448__rv_v3);
                        params_t191.add_input(ext_c1a_index_block_table);
                        params_t191.add_input(scores_inline24963);
                        params_t191.add_scalar(B_DYN);
                        params_t191.add_scalar(FWD_C1A_TABLE_DYN);
                        params_t191.add_scalar(attention_count_inline23434);
                        params_t191.add_scalar(width_inline25451);
                        params_t191.launch_spec.set_block_num(32);
                        TaskId params_t191_deps[1];
                        uint32_t params_t191_deps_count = 0;
                        params_t191_deps[params_t191_deps_count++] = score_tid_inline25450;
                        params_t191.set_dependencies(params_t191_deps, params_t191_deps_count);
                        TaskOutputTensors task_191_outs = rt_submit_aiv_task(116, params_t191);
                        TaskId short_tid_inline25454 = task_191_outs.task_id();
                        result_inline25453 = short_tid_inline25454;
                        topk_indices_inline23445__rv_v2 = topk_indices_inline23445__rv_v5;
                    } else {
                        int64_t leaf_count_inline25459 = ((width_inline25451 + 2047) / 2048);
                        int64_t arena_rows_inline25460 = (LOCAL_T_DYN * leaf_count_inline25459);
                        uint32_t pair_arena_inline25461_ci_shapes[2] = {static_cast<uint32_t>(arena_rows_inline25460), 1024};
                        TensorCreateInfo pair_arena_inline25461_ci(pair_arena_inline25461_ci_shapes, 2, DataType::FLOAT32);
                        TaskOutputTensors alloc_77 = alloc_tensors(pair_arena_inline25461_ci);
                        const Tensor& pair_arena_inline25461 = alloc_77.get_ref(0);
                        int64_t total_leaves_inline25462 = (static_cast<int64_t>(attention_count_inline23434) * leaf_count_inline25459);

                        // Spmd pool_topk_leaf_spmd: idx_topk_leaf_0
                        CoreTaskArgs params_t192;
                        params_t192.add_input(scores_inline24963);
                        params_t192.add_inout(pair_arena_inline25461);
                        params_t192.add_scalar(total_leaves_inline25462);
                        params_t192.add_scalar(leaf_count_inline25459);
                        params_t192.add_scalar(width_inline25451);
                        params_t192.launch_spec.set_block_num(32);
                        TaskId params_t192_deps[1];
                        uint32_t params_t192_deps_count = 0;
                        params_t192_deps[params_t192_deps_count++] = score_tid_inline25450;
                        params_t192.set_dependencies(params_t192_deps, params_t192_deps_count);
                        TaskOutputTensors task_192_outs = rt_submit_aiv_task(103, params_t192);
                        TaskId leaf_tid_inline25463 = task_192_outs.task_id();

                        // Spmd pool_topk_merge_spmd: pool_topk_merge
                        CoreTaskArgs params_t193;
                        params_t193.add_inout(pair_arena_inline25461);
                        params_t193.add_input(ext_token_to_req_indices);
                        params_t193.add_inout(topk_indices_inline23445__rv_v5);
                        params_t193.add_input(pool_block_ids_inline23448__rv_v3);
                        params_t193.add_input(ext_c1a_index_block_table);
                        params_t193.add_input(scores_inline24963);
                        params_t193.add_scalar(B_DYN);
                        params_t193.add_scalar(FWD_C1A_TABLE_DYN);
                        params_t193.add_scalar(attention_count_inline23434);
                        params_t193.add_scalar(leaf_count_inline25459);
                        params_t193.add_scalar(width_inline25451);
                        params_t193.launch_spec.set_block_num(32);
                        TaskId params_t193_deps[1];
                        uint32_t params_t193_deps_count = 0;
                        params_t193_deps[params_t193_deps_count++] = leaf_tid_inline25463;
                        params_t193.set_dependencies(params_t193_deps, params_t193_deps_count);
                        TaskOutputTensors task_193_outs = rt_submit_aiv_task(117, params_t193);
                        TaskId merge_tid_inline25471 = task_193_outs.task_id();
                        result_inline25453 = merge_tid_inline25471;
                        topk_indices_inline23445__rv_v2 = topk_indices_inline23445__rv_v5;
                    }
                    TaskId selection_ready_inline24967 = result_inline25453;
                    if ((static_cast<int64_t>(attention_count_inline23434) <= 12)) {
                        uint32_t selected_a_inline25649_ci_shapes[3] = {24, 640, 512};
                        TensorCreateInfo selected_a_inline25649_ci(selected_a_inline25649_ci_shapes, 3, DataType::BFLOAT16);
                        uint32_t selected_b_inline25650_ci_shapes[3] = {24, 640, 512};
                        TensorCreateInfo selected_b_inline25650_ci(selected_b_inline25650_ci_shapes, 3, DataType::BFLOAT16);
                        uint32_t combined_a_inline25651_ci_shapes[2] = {24, 640};
                        TensorCreateInfo combined_a_inline25651_ci(combined_a_inline25651_ci_shapes, 2, DataType::INT32);
                        uint32_t combined_b_inline25652_ci_shapes[2] = {24, 640};
                        TensorCreateInfo combined_b_inline25652_ci(combined_b_inline25652_ci_shapes, 2, DataType::INT32);
                        TaskOutputTensors alloc_78 = alloc_tensors(selected_a_inline25649_ci, selected_b_inline25650_ci, combined_a_inline25651_ci, combined_b_inline25652_ci);
                        const Tensor& selected_a_inline25649 = alloc_78.get_ref(0);
                        const Tensor& selected_b_inline25650 = alloc_78.get_ref(1);
                        const Tensor& combined_a_inline25651 = alloc_78.get_ref(2);
                        const Tensor& combined_b_inline25652 = alloc_78.get_ref(3);
                        TaskId done_a_inline25653 = selection_ready_inline24967;
                        TaskId done_b_inline25654 = selection_ready_inline24967;
                        TaskId done_a_inline25653__rv_v2 = done_a_inline25653;
                        TaskId done_b_inline25654__rv_v2 = done_b_inline25654;
                        Tensor combined_b_inline25652__rv_v2 = combined_b_inline25652;
                        for (int64_t start_inline25655 = 0; start_inline25655 < attention_count_inline23434; start_inline25655 += 48) {
                            int64_t width_inline25656 = std::min<int64_t>((((static_cast<int64_t>(attention_count_inline23434) - start_inline25655) + 1) / 2), 24);
                            int64_t active_a_inline25657 = width_inline25656;
                            uint32_t wi_a_inline25658_offsets[2] = {static_cast<uint32_t>(start_inline25655), 0};
                            uint32_t wi_a_inline25658_shapes[2] = {(wi_a_inline25658_offsets[0] >= ext_window_indices.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_a_inline25657), ext_window_indices.shapes[0] - wi_a_inline25658_offsets[0])), (wi_a_inline25658_offsets[1] >= ext_window_indices.shapes[1] ? 0u : std::min<uint32_t>(128, ext_window_indices.shapes[1] - wi_a_inline25658_offsets[1]))};
                            Tensor wi_a_inline25658 = ext_window_indices.view(wi_a_inline25658_shapes, wi_a_inline25658_offsets);
                            uint32_t ci_a_inline25659_offsets[2] = {static_cast<uint32_t>(start_inline25655), 0};
                            uint32_t ci_a_inline25659_shapes[2] = {(ci_a_inline25659_offsets[0] >= topk_indices_inline23445__rv_v2.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_a_inline25657), topk_indices_inline23445__rv_v2.shapes[0] - ci_a_inline25659_offsets[0])), (ci_a_inline25659_offsets[1] >= topk_indices_inline23445__rv_v2.shapes[1] ? 0u : std::min<uint32_t>(512, topk_indices_inline23445__rv_v2.shapes[1] - ci_a_inline25659_offsets[1]))};
                            Tensor ci_a_inline25659 = topk_indices_inline23445__rv_v2.view(ci_a_inline25659_shapes, ci_a_inline25659_offsets);
                            uint32_t query_a_inline25660_offsets[2] = {static_cast<uint32_t>(start_inline25655), 0};
                            uint32_t query_a_inline25660_shapes[2] = {(query_a_inline25660_offsets[0] >= query_inline24949__ssa_v1.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_a_inline25657), query_inline24949__ssa_v1.shapes[0] - query_a_inline25660_offsets[0])), (query_a_inline25660_offsets[1] >= query_inline24949__ssa_v1.shapes[1] ? 0u : std::min<uint32_t>(32768, query_inline24949__ssa_v1.shapes[1] - query_a_inline25660_offsets[1]))};
                            Tensor query_a_inline25660 = query_inline24949__ssa_v1.view(query_a_inline25660_shapes, query_a_inline25660_offsets);
                            uint32_t output_a_inline25661_offsets[2] = {static_cast<uint32_t>(start_inline25655), 0};
                            uint32_t output_a_inline25661_shapes[2] = {(output_a_inline25661_offsets[0] >= attended_inline24968.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_a_inline25657), attended_inline24968.shapes[0] - output_a_inline25661_offsets[0])), (output_a_inline25661_offsets[1] >= attended_inline24968.shapes[1] ? 0u : std::min<uint32_t>(32768, attended_inline24968.shapes[1] - output_a_inline25661_offsets[1]))};
                            Tensor output_a_inline25661 = attended_inline24968.view(output_a_inline25661_shapes, output_a_inline25661_offsets);
                            int64_t t__tmp_v1192 = (int64_t)layer_window_cache_inline23453.shapes[0];
                            int64_t window_rows_inline25672 = (t__tmp_v1192 * 128);
                            int64_t t__tmp_v1193 = (int64_t)layer_compressed_cache_inline23456.shapes[0];
                            int64_t compressed_rows_inline25673 = (t__tmp_v1193 * 128);
                            uint32_t w_inline25674_shapes[2] = {static_cast<uint32_t>(window_rows_inline25672), 512};
                            Tensor w_inline25674 = layer_window_cache_inline23453.reshape(w_inline25674_shapes, 2);
                            uint32_t ws_inline25675_shapes[2] = {static_cast<uint32_t>(window_rows_inline25672), 16};
                            Tensor ws_inline25675 = layer_window_cache_scale_inline23454.reshape(ws_inline25675_shapes, 2);
                            uint32_t c_inline25676_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline25673), 256};
                            Tensor c_inline25676 = layer_compressed_cache_inline23456.reshape(c_inline25676_shapes, 2);
                            uint32_t cs_inline25677_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline25673), 32};
                            Tensor cs_inline25677 = layer_compressed_cache_scale_inline23457.reshape(cs_inline25677_shapes, 2);
                            int64_t t__tmp_v1194 = 24;
                            int64_t selected_rows_inline25678 = (t__tmp_v1194 * 640);
                            uint32_t dst_inline25679_shapes[2] = {static_cast<uint32_t>(selected_rows_inline25678), 512};
                            Tensor dst_inline25679 = selected_a_inline25649.reshape(dst_inline25679_shapes, 2);
                            int64_t workers_inline25680 = std::max<int64_t>(std::min<int64_t>((static_cast<int64_t>(active_a_inline25657) * 20), 32), 1);

                            // Spmd c1a_gather_spmd_0: c1a_gather_0
                            CoreTaskArgs params_t194;
                            params_t194.add_input(wi_a_inline25658);
                            params_t194.add_input(w_inline25674);
                            params_t194.add_input(ws_inline25675);
                            params_t194.add_input(ci_a_inline25659);
                            params_t194.add_input(c_inline25676);
                            params_t194.add_input(cs_inline25677);
                            params_t194.add_inout(dst_inline25679);
                            params_t194.add_inout(combined_a_inline25651);
                            params_t194.add_scalar(active_a_inline25657);
                            params_t194.add_scalar(workers_inline25680);
                            params_t194.launch_spec.set_block_num(workers_inline25680);
                            TaskId params_t194_deps[1];
                            uint32_t params_t194_deps_count = 0;
                            if (done_a_inline25653__rv_v2.is_valid()) params_t194_deps[params_t194_deps_count++] = done_a_inline25653__rv_v2;
                            params_t194.set_dependencies(params_t194_deps, params_t194_deps_count);
                            TaskOutputTensors task_194_outs = rt_submit_aiv_task(57, params_t194);
                            TaskId gathered_inline25681 = task_194_outs.task_id();
                            TaskId gathered_a_inline25663 = gathered_inline25681;
                            int64_t tokens_inline25719 = (int64_t)query_a_inline25660.shapes[0];
                            int64_t head_rows_inline25720 = (tokens_inline25719 * 64);
                            uint32_t query_flat_inline25721_shapes[2] = {static_cast<uint32_t>(head_rows_inline25720), 512};
                            Tensor query_flat_inline25721 = query_a_inline25660.reshape(query_flat_inline25721_shapes, 2);
                            uint32_t output_flat_inline25722_shapes[2] = {static_cast<uint32_t>(head_rows_inline25720), 512};
                            Tensor output_flat_inline25722 = output_a_inline25661.reshape(output_flat_inline25722_shapes, 2);
                            uint32_t key_flat_inline25723_shapes[2] = {15360, 512};
                            Tensor key_flat_inline25723 = selected_a_inline25649.reshape(key_flat_inline25723_shapes, 2);
                            int64_t attend_tasks_inline25724 = (static_cast<int64_t>(active_a_inline25657) * 2);
                            int64_t workers_inline25725 = std::min<int64_t>(attend_tasks_inline25724, 16);
                            int32_t t__tmp_v1199 = rt_available_cluster_count();
                            int64_t workers_inline25725__ssa_v3 = std::min<int64_t>(workers_inline25725, static_cast<int64_t>(t__tmp_v1199));
                            TaskId ret__tmp_v0_guarded_0_41 = TaskId::invalid();
                            if ((0 < workers_inline25725__ssa_v3)) {

                                // Group c1a_sparse_attn: MixedKernels (AIC + AIV lanes)
                                CoreTaskArgs params_t195;
                                params_t195.add_inout(output_flat_inline25722);
                                params_t195.add_input(query_flat_inline25721);
                                params_t195.add_input(key_flat_inline25723);
                                params_t195.add_input(combined_a_inline25651);
                                params_t195.add_input(layer_sink_inline23470);
                                params_t195.add_scalar(attend_tasks_inline25724);
                                params_t195.add_scalar(workers_inline25725__ssa_v3);
                                MixedKernels mixed_195 = {58, 59, INVALID_KERNEL_ID};
                                params_t195.launch_spec.set_block_num(workers_inline25725__ssa_v3);
                                params_t195.launch_spec.set_require_sync_start(true);
                                TaskId params_t195_deps[1];
                                uint32_t params_t195_deps_count = 0;
                                params_t195_deps[params_t195_deps_count++] = gathered_inline25681;
                                params_t195.set_dependencies(params_t195_deps, params_t195_deps_count);
                                TaskOutputTensors task_195_outs = rt_submit_task(mixed_195, params_t195);
                                TaskId ret__tmp_v0_nonempty_0_46 = task_195_outs.task_id();
                                ret__tmp_v0_guarded_0_41 = ret__tmp_v0_nonempty_0_46;
                            } else {

                                // Phase-fence barrier 39: dependency-only dummy task
                                CoreTaskArgs params_phase_fence_barrier_39;
                                TaskId params_phase_fence_barrier_39_deps[1];
                                uint32_t params_phase_fence_barrier_39_deps_count = 0;
                                params_phase_fence_barrier_39_deps[params_phase_fence_barrier_39_deps_count++] = gathered_inline25681;
                                params_phase_fence_barrier_39.set_dependencies(params_phase_fence_barrier_39_deps, params_phase_fence_barrier_39_deps_count);
                                TaskId spmd_empty_tid_32 = TaskId::invalid();
                                if (params_phase_fence_barrier_39_deps_count > 0) {
                                    TaskOutputTensors phase_fence_barrier_39_outs = rt_submit_dummy_task(params_phase_fence_barrier_39);
                                    spmd_empty_tid_32 = phase_fence_barrier_39_outs.task_id();
                                }
                                ret__tmp_v0_guarded_0_41 = spmd_empty_tid_32;
                            }
                            TaskId attend_tid_inline25726 = ret__tmp_v0_guarded_0_41;
                            TaskId done_a_inline25653__ssa_v3 = attend_tid_inline25726;
                            TaskId done_b_inline25654__phi_v4 = TaskId::invalid();
                            Tensor combined_b_inline25652__rv_v1 = ext_engram_lookup;
                            if (((start_inline25655 + width_inline25656) < static_cast<int64_t>(attention_count_inline23434))) {
                                int64_t active_b_inline25664 = std::min<int64_t>(((static_cast<int64_t>(attention_count_inline23434) - start_inline25655) - width_inline25656), 24);
                                uint32_t wi_b_inline25665_offsets[2] = {static_cast<uint32_t>((start_inline25655 + width_inline25656)), 0};
                                uint32_t wi_b_inline25665_shapes[2] = {(wi_b_inline25665_offsets[0] >= ext_window_indices.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_b_inline25664), ext_window_indices.shapes[0] - wi_b_inline25665_offsets[0])), (wi_b_inline25665_offsets[1] >= ext_window_indices.shapes[1] ? 0u : std::min<uint32_t>(128, ext_window_indices.shapes[1] - wi_b_inline25665_offsets[1]))};
                                Tensor wi_b_inline25665 = ext_window_indices.view(wi_b_inline25665_shapes, wi_b_inline25665_offsets);
                                uint32_t ci_b_inline25666_offsets[2] = {static_cast<uint32_t>((start_inline25655 + width_inline25656)), 0};
                                uint32_t ci_b_inline25666_shapes[2] = {(ci_b_inline25666_offsets[0] >= topk_indices_inline23445__rv_v2.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_b_inline25664), topk_indices_inline23445__rv_v2.shapes[0] - ci_b_inline25666_offsets[0])), (ci_b_inline25666_offsets[1] >= topk_indices_inline23445__rv_v2.shapes[1] ? 0u : std::min<uint32_t>(512, topk_indices_inline23445__rv_v2.shapes[1] - ci_b_inline25666_offsets[1]))};
                                Tensor ci_b_inline25666 = topk_indices_inline23445__rv_v2.view(ci_b_inline25666_shapes, ci_b_inline25666_offsets);
                                uint32_t query_b_inline25667_offsets[2] = {static_cast<uint32_t>((start_inline25655 + width_inline25656)), 0};
                                uint32_t query_b_inline25667_shapes[2] = {(query_b_inline25667_offsets[0] >= query_inline24949__ssa_v1.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_b_inline25664), query_inline24949__ssa_v1.shapes[0] - query_b_inline25667_offsets[0])), (query_b_inline25667_offsets[1] >= query_inline24949__ssa_v1.shapes[1] ? 0u : std::min<uint32_t>(32768, query_inline24949__ssa_v1.shapes[1] - query_b_inline25667_offsets[1]))};
                                Tensor query_b_inline25667 = query_inline24949__ssa_v1.view(query_b_inline25667_shapes, query_b_inline25667_offsets);
                                uint32_t output_b_inline25668_offsets[2] = {static_cast<uint32_t>((start_inline25655 + width_inline25656)), 0};
                                uint32_t output_b_inline25668_shapes[2] = {(output_b_inline25668_offsets[0] >= attended_inline24968.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_b_inline25664), attended_inline24968.shapes[0] - output_b_inline25668_offsets[0])), (output_b_inline25668_offsets[1] >= attended_inline24968.shapes[1] ? 0u : std::min<uint32_t>(32768, attended_inline24968.shapes[1] - output_b_inline25668_offsets[1]))};
                                Tensor output_b_inline25668 = attended_inline24968.view(output_b_inline25668_shapes, output_b_inline25668_offsets);
                                int64_t t__tmp_v1218 = (int64_t)layer_window_cache_inline23453.shapes[0];
                                int64_t window_rows_inline25760 = (t__tmp_v1218 * 128);
                                int64_t t__tmp_v1219 = (int64_t)layer_compressed_cache_inline23456.shapes[0];
                                int64_t compressed_rows_inline25761 = (t__tmp_v1219 * 128);
                                uint32_t w_inline25762_shapes[2] = {static_cast<uint32_t>(window_rows_inline25760), 512};
                                Tensor w_inline25762 = layer_window_cache_inline23453.reshape(w_inline25762_shapes, 2);
                                uint32_t ws_inline25763_shapes[2] = {static_cast<uint32_t>(window_rows_inline25760), 16};
                                Tensor ws_inline25763 = layer_window_cache_scale_inline23454.reshape(ws_inline25763_shapes, 2);
                                uint32_t c_inline25764_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline25761), 256};
                                Tensor c_inline25764 = layer_compressed_cache_inline23456.reshape(c_inline25764_shapes, 2);
                                uint32_t cs_inline25765_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline25761), 32};
                                Tensor cs_inline25765 = layer_compressed_cache_scale_inline23457.reshape(cs_inline25765_shapes, 2);
                                int64_t t__tmp_v1220 = 24;
                                int64_t selected_rows_inline25766 = (t__tmp_v1220 * 640);
                                uint32_t dst_inline25767_shapes[2] = {static_cast<uint32_t>(selected_rows_inline25766), 512};
                                Tensor dst_inline25767 = selected_b_inline25650.reshape(dst_inline25767_shapes, 2);
                                int64_t workers_inline25768 = std::max<int64_t>(std::min<int64_t>((static_cast<int64_t>(active_b_inline25664) * 20), 32), 1);

                                // Spmd c1a_gather_spmd_1: c1a_gather_0
                                CoreTaskArgs params_t196;
                                params_t196.add_input(wi_b_inline25665);
                                params_t196.add_input(w_inline25762);
                                params_t196.add_input(ws_inline25763);
                                params_t196.add_input(ci_b_inline25666);
                                params_t196.add_input(c_inline25764);
                                params_t196.add_input(cs_inline25765);
                                params_t196.add_inout(dst_inline25767);
                                params_t196.add_inout(combined_b_inline25652__rv_v2);
                                params_t196.add_scalar(active_b_inline25664);
                                params_t196.add_scalar(workers_inline25768);
                                params_t196.launch_spec.set_block_num(workers_inline25768);
                                TaskId params_t196_deps[1];
                                uint32_t params_t196_deps_count = 0;
                                if (done_b_inline25654__rv_v2.is_valid()) params_t196_deps[params_t196_deps_count++] = done_b_inline25654__rv_v2;
                                params_t196.set_dependencies(params_t196_deps, params_t196_deps_count);
                                TaskOutputTensors task_196_outs = rt_submit_aiv_task(57, params_t196);
                                TaskId gathered_inline25769 = task_196_outs.task_id();
                                TaskId gathered_b_inline25670 = gathered_inline25769;
                                int64_t tokens_inline25807 = (int64_t)query_b_inline25667.shapes[0];
                                int64_t head_rows_inline25808 = (tokens_inline25807 * 64);
                                uint32_t query_flat_inline25809_shapes[2] = {static_cast<uint32_t>(head_rows_inline25808), 512};
                                Tensor query_flat_inline25809 = query_b_inline25667.reshape(query_flat_inline25809_shapes, 2);
                                uint32_t output_flat_inline25810_shapes[2] = {static_cast<uint32_t>(head_rows_inline25808), 512};
                                Tensor output_flat_inline25810 = output_b_inline25668.reshape(output_flat_inline25810_shapes, 2);
                                uint32_t key_flat_inline25811_shapes[2] = {15360, 512};
                                Tensor key_flat_inline25811 = selected_b_inline25650.reshape(key_flat_inline25811_shapes, 2);
                                int64_t attend_tasks_inline25812 = (static_cast<int64_t>(active_b_inline25664) * 2);
                                int64_t workers_inline25813 = std::min<int64_t>(attend_tasks_inline25812, 16);
                                int32_t t__tmp_v1225 = rt_available_cluster_count();
                                int64_t workers_inline25813__ssa_v3 = std::min<int64_t>(workers_inline25813, static_cast<int64_t>(t__tmp_v1225));
                                TaskId ret__tmp_v0_guarded_0_42 = TaskId::invalid();
                                if ((0 < workers_inline25813__ssa_v3)) {

                                    // Group c1a_sparse_attn: MixedKernels (AIC + AIV lanes)
                                    CoreTaskArgs params_t197;
                                    params_t197.add_inout(output_flat_inline25810);
                                    params_t197.add_input(query_flat_inline25809);
                                    params_t197.add_input(key_flat_inline25811);
                                    params_t197.add_input(combined_b_inline25652__rv_v2);
                                    params_t197.add_input(layer_sink_inline23470);
                                    params_t197.add_scalar(attend_tasks_inline25812);
                                    params_t197.add_scalar(workers_inline25813__ssa_v3);
                                    MixedKernels mixed_197 = {58, 59, INVALID_KERNEL_ID};
                                    params_t197.launch_spec.set_block_num(workers_inline25813__ssa_v3);
                                    params_t197.launch_spec.set_require_sync_start(true);
                                    TaskId params_t197_deps[1];
                                    uint32_t params_t197_deps_count = 0;
                                    params_t197_deps[params_t197_deps_count++] = gathered_inline25769;
                                    params_t197.set_dependencies(params_t197_deps, params_t197_deps_count);
                                    TaskOutputTensors task_197_outs = rt_submit_task(mixed_197, params_t197);
                                    TaskId ret__tmp_v0_nonempty_0_47 = task_197_outs.task_id();
                                    ret__tmp_v0_guarded_0_42 = ret__tmp_v0_nonempty_0_47;
                                } else {

                                    // Phase-fence barrier 40: dependency-only dummy task
                                    CoreTaskArgs params_phase_fence_barrier_40;
                                    TaskId params_phase_fence_barrier_40_deps[1];
                                    uint32_t params_phase_fence_barrier_40_deps_count = 0;
                                    params_phase_fence_barrier_40_deps[params_phase_fence_barrier_40_deps_count++] = gathered_inline25769;
                                    params_phase_fence_barrier_40.set_dependencies(params_phase_fence_barrier_40_deps, params_phase_fence_barrier_40_deps_count);
                                    TaskId spmd_empty_tid_33 = TaskId::invalid();
                                    if (params_phase_fence_barrier_40_deps_count > 0) {
                                        TaskOutputTensors phase_fence_barrier_40_outs = rt_submit_dummy_task(params_phase_fence_barrier_40);
                                        spmd_empty_tid_33 = phase_fence_barrier_40_outs.task_id();
                                    }
                                    ret__tmp_v0_guarded_0_42 = spmd_empty_tid_33;
                                }
                                TaskId attend_tid_inline25814 = ret__tmp_v0_guarded_0_42;
                                TaskId done_b_inline25654__ssa_v3 = attend_tid_inline25814;
                                done_b_inline25654__phi_v4 = done_b_inline25654__ssa_v3;
                                combined_b_inline25652__rv_v1 = combined_b_inline25652__rv_v2;
                            } else {
                                done_b_inline25654__phi_v4 = done_b_inline25654__rv_v2;
                                combined_b_inline25652__rv_v1 = combined_b_inline25652__rv_v2;
                            }
                            done_a_inline25653__rv_v2 = done_a_inline25653__ssa_v3;
                            done_b_inline25654__rv_v2 = done_b_inline25654__phi_v4;
                            combined_b_inline25652__rv_v2 = combined_b_inline25652__rv_v1;
                        }

                        // Phase-fence barrier 41: dependency-only dummy task
                        CoreTaskArgs params_phase_fence_barrier_41;
                        TaskId params_phase_fence_barrier_41_deps[2];
                        uint32_t params_phase_fence_barrier_41_deps_count = 0;
                        if (done_a_inline25653__rv_v2.is_valid()) params_phase_fence_barrier_41_deps[params_phase_fence_barrier_41_deps_count++] = done_a_inline25653__rv_v2;
                        if (done_b_inline25654__rv_v2.is_valid()) params_phase_fence_barrier_41_deps[params_phase_fence_barrier_41_deps_count++] = done_b_inline25654__rv_v2;
                        params_phase_fence_barrier_41.set_dependencies(params_phase_fence_barrier_41_deps, params_phase_fence_barrier_41_deps_count);
                        TaskId complete_inline25671 = TaskId::invalid();
                        if (params_phase_fence_barrier_41_deps_count > 0) {
                            TaskOutputTensors phase_fence_barrier_41_outs = rt_submit_dummy_task(params_phase_fence_barrier_41);
                            complete_inline25671 = phase_fence_barrier_41_outs.task_id();
                        }
                    } else {
                        uint32_t selected_inline24969_ci_shapes[3] = {static_cast<uint32_t>(LOCAL_T_DYN), 640, 512};
                        TensorCreateInfo selected_inline24969_ci(selected_inline24969_ci_shapes, 3, DataType::BFLOAT16);
                        uint32_t indices_inline24970_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 640};
                        TensorCreateInfo indices_inline24970_ci(indices_inline24970_ci_shapes, 2, DataType::INT32);
                        TaskOutputTensors alloc_79 = alloc_tensors(selected_inline24969_ci, indices_inline24970_ci);
                        const Tensor& selected_inline24969 = alloc_79.get_ref(0);
                        const Tensor& indices_inline24970 = alloc_79.get_ref(1);
                        int64_t t__tmp_v1244 = (int64_t)layer_window_cache_inline23453.shapes[0];
                        int64_t window_rows_inline25848 = (t__tmp_v1244 * 128);
                        int64_t t__tmp_v1245 = (int64_t)layer_compressed_cache_inline23456.shapes[0];
                        int64_t compressed_rows_inline25849 = (t__tmp_v1245 * 128);
                        uint32_t w_inline25850_shapes[2] = {static_cast<uint32_t>(window_rows_inline25848), 512};
                        Tensor w_inline25850 = layer_window_cache_inline23453.reshape(w_inline25850_shapes, 2);
                        uint32_t ws_inline25851_shapes[2] = {static_cast<uint32_t>(window_rows_inline25848), 16};
                        Tensor ws_inline25851 = layer_window_cache_scale_inline23454.reshape(ws_inline25851_shapes, 2);
                        uint32_t c_inline25852_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline25849), 256};
                        Tensor c_inline25852 = layer_compressed_cache_inline23456.reshape(c_inline25852_shapes, 2);
                        uint32_t cs_inline25853_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline25849), 32};
                        Tensor cs_inline25853 = layer_compressed_cache_scale_inline23457.reshape(cs_inline25853_shapes, 2);
                        int64_t selected_rows_inline25854 = (LOCAL_T_DYN * 640);
                        uint32_t dst_inline25855_shapes[2] = {static_cast<uint32_t>(selected_rows_inline25854), 512};
                        Tensor dst_inline25855 = selected_inline24969.reshape(dst_inline25855_shapes, 2);
                        int64_t workers_inline25856 = std::max<int64_t>(std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) * 20), 64), 1);

                        // Spmd c1a_gather_spmd_2: c1a_gather
                        CoreTaskArgs params_t198;
                        params_t198.add_input(ext_window_indices);
                        params_t198.add_input(w_inline25850);
                        params_t198.add_input(ws_inline25851);
                        params_t198.add_input(topk_indices_inline23445__rv_v2);
                        params_t198.add_input(c_inline25852);
                        params_t198.add_input(cs_inline25853);
                        params_t198.add_inout(dst_inline25855);
                        params_t198.add_inout(indices_inline24970);
                        params_t198.add_scalar(attention_count_inline23434);
                        params_t198.add_scalar(workers_inline25856);
                        params_t198.launch_spec.set_block_num(workers_inline25856);
                        TaskId params_t198_deps[1];
                        uint32_t params_t198_deps_count = 0;
                        if (result_inline25453.is_valid()) params_t198_deps[params_t198_deps_count++] = result_inline25453;
                        params_t198.set_dependencies(params_t198_deps, params_t198_deps_count);
                        TaskOutputTensors task_198_outs = rt_submit_aiv_task(60, params_t198);
                        TaskId gathered_inline25857 = task_198_outs.task_id();
                        TaskId gather_tid_inline24974 = gathered_inline25857;
                        int64_t head_rows_inline25896 = (LOCAL_T_DYN * 64);
                        int64_t cache_rows_inline25897 = (LOCAL_T_DYN * 640);
                        uint32_t qflat_inline25898_shapes[2] = {static_cast<uint32_t>(head_rows_inline25896), 512};
                        Tensor qflat_inline25898 = query_inline24949__ssa_v1.reshape(qflat_inline25898_shapes, 2);
                        uint32_t kflat_inline25899_shapes[2] = {static_cast<uint32_t>(cache_rows_inline25897), 512};
                        Tensor kflat_inline25899 = selected_inline24969.reshape(kflat_inline25899_shapes, 2);
                        uint32_t oflat_inline25900_shapes[2] = {static_cast<uint32_t>(head_rows_inline25896), 512};
                        Tensor oflat_inline25900 = attended_inline24968.reshape(oflat_inline25900_shapes, 2);
                        int64_t workers_inline25901 = std::max<int64_t>(std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) * 2), 32), 1);

                        // Group sparse_online_cube: MixedKernels (AIC + AIV lanes)
                        CoreTaskArgs params_t199;
                        params_t199.add_inout(oflat_inline25900);
                        params_t199.add_input(qflat_inline25898);
                        params_t199.add_input(kflat_inline25899);
                        params_t199.add_input(indices_inline24970);
                        params_t199.add_input(layer_sink_inline23470);
                        params_t199.add_scalar(attention_count_inline23434);
                        params_t199.add_scalar(workers_inline25901);
                        MixedKernels mixed_199 = {61, 62, INVALID_KERNEL_ID};
                        params_t199.launch_spec.set_block_num(workers_inline25901);
                        TaskId params_t199_deps[1];
                        uint32_t params_t199_deps_count = 0;
                        params_t199_deps[params_t199_deps_count++] = gathered_inline25857;
                        params_t199.set_dependencies(params_t199_deps, params_t199_deps_count);
                        TaskOutputTensors task_199_outs = rt_submit_task(mixed_199, params_t199);
                    }
                    int64_t t_linear_inline25946 = (((LOCAL_T_DYN + 15) / 16) * 16);
                    uint32_t x_flat_inline25947_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                    Tensor x_flat_inline25947 = ext_x_hc.reshape(x_flat_inline25947_shapes, 2);
                    uint32_t inv_rms_inline25948_ci_shapes[2] = {static_cast<uint32_t>(t_linear_inline25946), 1};
                    TensorCreateInfo inv_rms_inline25948_ci(inv_rms_inline25948_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_80 = alloc_tensors(inv_rms_inline25948_ci);
                    const Tensor& inv_rms_inline25948 = alloc_80.get_ref(0);
                    int64_t spmd_core_num_20 = ((LOCAL_T_DYN + 7) / 8);
                    Tensor inv_rms_inline25948__ssa_v1 = ext_engram_lookup;
                    if ((0 < spmd_core_num_20)) {

                        // Spmd mhc_rms_spmd_4: mhc_rms
                        CoreTaskArgs params_t200;
                        params_t200.add_input(x_flat_inline25947);
                        params_t200.add_inout(inv_rms_inline25948);
                        params_t200.add_scalar(LOCAL_T_DYN);
                        params_t200.launch_spec.set_block_num(spmd_core_num_20);
                        rt_submit_aiv_task(28, params_t200);
                        inv_rms_inline25948__ssa_v1 = inv_rms_inline25948;
                    } else {
                        inv_rms_inline25948__ssa_v1 = inv_rms_inline25948;
                    }
                    uint32_t mixes_partials_inline25957_ci_shapes[2] = {static_cast<uint32_t>((t_linear_inline25946 * 4)), 32};
                    TensorCreateInfo mixes_partials_inline25957_ci(mixes_partials_inline25957_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_81 = alloc_tensors(mixes_partials_inline25957_ci);
                    const Tensor& mixes_partials_inline25957 = alloc_81.get_ref(0);
                    int64_t tasks_inline25958 = (((LOCAL_T_DYN + 79) / 80) * 4);
                    int32_t t__tmp_v1267 = rt_available_cluster_count();
                    int64_t workers_inline25959 = std::min<int64_t>(tasks_inline25958, static_cast<int64_t>(t__tmp_v1267));
                    if ((0 < workers_inline25959)) {

                        // Spmd mhc_linear_4: _hc_linear
                        CoreTaskArgs params_t201;
                        params_t201.add_input(x_flat_inline25947);
                        params_t201.add_input(layer_hc_fn_inline23458);
                        params_t201.add_inout(mixes_partials_inline25957);
                        params_t201.add_scalar(LOCAL_T_DYN);
                        params_t201.add_scalar(t_linear_inline25946);
                        params_t201.launch_spec.set_block_num(workers_inline25959);
                        rt_submit_aic_task(29, params_t201);
                    } else {
                    }
                    Tensor inv_rms_inline24976 = inv_rms_inline25948__ssa_v1;
                    Tensor partials_inline24977 = mixes_partials_inline25957;
                    int64_t linear_tokens_inline25969 = (int64_t)inv_rms_inline24976.shapes[0];
                    uint32_t base_row_inline25970_shapes[2] = {1, 24};
                    Tensor base_row_inline25970 = layer_hc_base_inline23460.reshape(base_row_inline25970_shapes, 2);
                    uint32_t residual_flat_inline25971_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 16};
                    Tensor residual_flat_inline25971 = residual_mix_inline24945.reshape(residual_flat_inline25971_shapes, 2);
                    uint32_t indices_scale0_inline25972[1] = {0};
                    float scale0_inline25972 = get_tensor_data<float>(layer_hc_scale_inline23459, 1, indices_scale0_inline25972);
                    uint32_t indices_scale1_inline25973[1] = {1};
                    float scale1_inline25973 = get_tensor_data<float>(layer_hc_scale_inline23459, 1, indices_scale1_inline25973);
                    uint32_t indices_scale2_inline25974[1] = {2};
                    float scale2_inline25974 = get_tensor_data<float>(layer_hc_scale_inline23459, 1, indices_scale2_inline25974);
                    int64_t blocks_inline25975 = ((LOCAL_T_DYN + 1) / 2);
                    int64_t workers_inline25976 = std::min<int64_t>(blocks_inline25975, 12);
                    Tensor ret__tmp_v0_guarded_0_43 = ext_engram_lookup;
                    Tensor ret__tmp_v0_guarded_1_31 = ext_engram_lookup;
                    if ((0 < workers_inline25976)) {

                        // Spmd mhc_sinkhorn_spmd_4: mhc_sinkhorn
                        CoreTaskArgs params_t202;
                        params_t202.add_input(partials_inline24977);
                        params_t202.add_input(inv_rms_inline24976);
                        params_t202.add_input(base_row_inline25970);
                        params_t202.add_inout(attention_pre_mix_inline23444__rv_v3);
                        params_t202.add_inout(post_mix_inline24944);
                        params_t202.add_inout(residual_flat_inline25971);
                        params_t202.add_scalar(LOCAL_T_DYN);
                        params_t202.add_scalar(blocks_inline25975);
                        params_t202.add_scalar(workers_inline25976);
                        params_t202.add_scalar(linear_tokens_inline25969);
                        params_t202.add_scalar(to_u64(scale0_inline25972));
                        params_t202.add_scalar(to_u64(scale1_inline25973));
                        params_t202.add_scalar(to_u64(scale2_inline25974));
                        params_t202.launch_spec.set_block_num(workers_inline25976);
                        TaskOutputTensors task_202_outs = rt_submit_aiv_task(30, params_t202);
                        ret__tmp_v0_guarded_0_43 = attention_pre_mix_inline23444__rv_v3;
                        ret__tmp_v0_guarded_1_31 = post_mix_inline24944;
                    } else {

                        // Phase-fence barrier 42: dependency-only dummy task
                        CoreTaskArgs params_phase_fence_barrier_42;
                        TaskOutputTensors phase_fence_barrier_42_outs = rt_submit_dummy_task(params_phase_fence_barrier_42);
                        TaskId spmd_empty_tid_34 = phase_fence_barrier_42_outs.task_id();
                        ret__tmp_v0_guarded_0_43 = attention_pre_mix_inline23444__rv_v3;
                        ret__tmp_v0_guarded_1_31 = post_mix_inline24944;
                    }
                    Tensor attention_pre_mix_inline23444__ssa_v2 = ret__tmp_v0_guarded_0_43;
                    Tensor post_mix_inline24944__ssa_v1 = ret__tmp_v0_guarded_1_31;
                    for (int64_t group_start_inline26081 = 0; group_start_inline26081 < LOCAL_T_DYN; group_start_inline26081 += 80) {
                        uint32_t payload_all_inline26091_ci_shapes[2] = {640, 4096};
                        TensorCreateInfo payload_all_inline26091_ci(payload_all_inline26091_ci_shapes, 2, DataType::FP8E4M3FN);
                        uint32_t scales_all_inline26092_ci_shapes[2] = {8, 10240};
                        TensorCreateInfo scales_all_inline26092_ci(scales_all_inline26092_ci_shapes, 2, DataType::FP8E8M0);
                        uint32_t projected_inline26169_ci_shapes[2] = {80, 8192};
                        TensorCreateInfo projected_inline26169_ci(projected_inline26169_ci_shapes, 2, DataType::BFLOAT16);
                        uint32_t payload_inline26172_ci_shapes[2] = {80, 8192};
                        TensorCreateInfo payload_inline26172_ci(payload_inline26172_ci_shapes, 2, DataType::FP8E4M3FN);
                        uint32_t activation_scale_backing_inline26173_ci_shapes[2] = {1, 20480};
                        TensorCreateInfo activation_scale_backing_inline26173_ci(activation_scale_backing_inline26173_ci_shapes, 2, DataType::FP8E8M0);
                        TaskOutputTensors alloc_82 = alloc_tensors(payload_all_inline26091_ci, scales_all_inline26092_ci, projected_inline26169_ci, payload_inline26172_ci, activation_scale_backing_inline26173_ci);
                        const Tensor& payload_all_inline26091 = alloc_82.get_ref(0);
                        const Tensor& scales_all_inline26092 = alloc_82.get_ref(1);
                        const Tensor& projected_inline26169 = alloc_82.get_ref(2);
                        const Tensor& payload_inline26172 = alloc_82.get_ref(3);
                        const Tensor& activation_scale_backing_inline26173 = alloc_82.get_ref(4);
                        int64_t group_rows_inline26082 = std::min<int64_t>((LOCAL_T_DYN - group_start_inline26081), 80);
                        int64_t active_rows_inline26083 = std::max<int64_t>(std::min<int64_t>(group_rows_inline26082, (static_cast<int64_t>(attention_count_inline23434) - group_start_inline26081)), 0);
                        uint32_t attended_group_inline26084_offsets[2] = {static_cast<uint32_t>(group_start_inline26081), 0};
                        uint32_t attended_group_inline26084_shapes[2] = {(attended_group_inline26084_offsets[0] >= attended_inline24968.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline26082), attended_inline24968.shapes[0] - attended_group_inline26084_offsets[0])), (attended_group_inline26084_offsets[1] >= attended_inline24968.shapes[1] ? 0u : std::min<uint32_t>(32768, attended_inline24968.shapes[1] - attended_group_inline26084_offsets[1]))};
                        Tensor attended_group_inline26084 = attended_inline24968.view(attended_group_inline26084_shapes, attended_group_inline26084_offsets);
                        uint32_t cosine_group_inline26085_offsets[2] = {static_cast<uint32_t>(group_start_inline26081), 0};
                        uint32_t cosine_group_inline26085_shapes[2] = {(cosine_group_inline26085_offsets[0] >= compressed_token_rope_cos_inline23437.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline26082), compressed_token_rope_cos_inline23437.shapes[0] - cosine_group_inline26085_offsets[0])), (cosine_group_inline26085_offsets[1] >= compressed_token_rope_cos_inline23437.shapes[1] ? 0u : std::min<uint32_t>(32, compressed_token_rope_cos_inline23437.shapes[1] - cosine_group_inline26085_offsets[1]))};
                        Tensor cosine_group_inline26085 = compressed_token_rope_cos_inline23437.view(cosine_group_inline26085_shapes, cosine_group_inline26085_offsets);
                        uint32_t sine_group_inline26086_offsets[2] = {static_cast<uint32_t>(group_start_inline26081), 0};
                        uint32_t sine_group_inline26086_shapes[2] = {(sine_group_inline26086_offsets[0] >= compressed_token_rope_sin_inline23438.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline26082), compressed_token_rope_sin_inline23438.shapes[0] - sine_group_inline26086_offsets[0])), (sine_group_inline26086_offsets[1] >= compressed_token_rope_sin_inline23438.shapes[1] ? 0u : std::min<uint32_t>(32, compressed_token_rope_sin_inline23438.shapes[1] - sine_group_inline26086_offsets[1]))};
                        Tensor sine_group_inline26086 = compressed_token_rope_sin_inline23438.view(sine_group_inline26086_shapes, sine_group_inline26086_offsets);
                        uint32_t output_group_inline26087_offsets[2] = {static_cast<uint32_t>(group_start_inline26081), 0};
                        uint32_t output_group_inline26087_shapes[2] = {(output_group_inline26087_offsets[0] >= attn_out_inline24978.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline26082), attn_out_inline24978.shapes[0] - output_group_inline26087_offsets[0])), (output_group_inline26087_offsets[1] >= attn_out_inline24978.shapes[1] ? 0u : std::min<uint32_t>(5120, attn_out_inline24978.shapes[1] - output_group_inline26087_offsets[1]))};
                        Tensor output_group_inline26087 = attn_out_inline24978.view(output_group_inline26087_shapes, output_group_inline26087_offsets);
                        int64_t t__tmp_v1268 = (int64_t)attended_group_inline26084.shapes[0];
                        int64_t last_input_row_inline26090 = std::max<int64_t>((t__tmp_v1268 - 1), 0);
                        int32_t t__tmp_v1269 = rt_available_cluster_count();
                        int64_t quant_workers_inline26095 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v1269) * 2), 64);
                        Tensor ret__tmp_v0_guarded_0_44 = ext_engram_lookup;
                        Tensor ret__tmp_v0_guarded_1_32 = ext_engram_lookup;
                        if ((0 < quant_workers_inline26095)) {

                            // Spmd rope_quant_o_pooled_spmd_3: rope_quant_o_pooled
                            CoreTaskArgs params_t203;
                            params_t203.add_inout(payload_all_inline26091);
                            params_t203.add_inout(scales_all_inline26092);
                            params_t203.add_input(cosine_group_inline26085);
                            params_t203.add_input(sine_group_inline26086);
                            params_t203.add_input(attended_group_inline26084);
                            params_t203.add_scalar(quant_workers_inline26095);
                            params_t203.add_scalar(active_rows_inline26083);
                            params_t203.add_scalar(last_input_row_inline26090);
                            params_t203.launch_spec.set_block_num(quant_workers_inline26095);
                            rt_submit_aiv_task(31, params_t203);
                            ret__tmp_v0_guarded_0_44 = payload_all_inline26091;
                            ret__tmp_v0_guarded_1_32 = scales_all_inline26092;
                        } else {
                            ret__tmp_v0_guarded_0_44 = payload_all_inline26091;
                            ret__tmp_v0_guarded_1_32 = scales_all_inline26092;
                        }
                        Tensor payload_all_inline26091__rv_v2 = ret__tmp_v0_guarded_0_44;
                        Tensor scales_all_inline26092__rv_v2 = ret__tmp_v0_guarded_1_32;
                        int32_t t__tmp_v1272 = rt_available_cluster_count();
                        int64_t oa_workers_inline26170 = std::min<int64_t>(static_cast<int64_t>(t__tmp_v1272), 32);
                        int64_t oa_workers_v1_inline26171 = std::max<int64_t>(oa_workers_inline26170, 1);

                        // Spmd proj_o_a_3: _o_proj_a
                        CoreTaskArgs params_t204;
                        params_t204.add_input(payload_all_inline26091__rv_v2);
                        params_t204.add_input(layer_wo_a_inline23471);
                        params_t204.add_input(scales_all_inline26092__rv_v2);
                        params_t204.add_input(layer_wo_a_scale_inline23472);
                        params_t204.add_inout(projected_inline26169);
                        params_t204.add_scalar((uint64_t)8192);
                        params_t204.launch_spec.set_block_num(oa_workers_v1_inline26171);
                        rt_submit_aic_task(32, params_t204);
                        int32_t t__tmp_v1273 = rt_available_cluster_count();
                        int64_t output_workers_inline26175 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v1273) * 2), 64);
                        Tensor ret__tmp_v0_guarded_0_45 = ext_engram_lookup;
                        Tensor ret__tmp_v0_guarded_1_33 = ext_engram_lookup;
                        if ((0 < output_workers_inline26175)) {

                            // Spmd quant_o_a_spmd_3: quant_o_a
                            CoreTaskArgs params_t205;
                            params_t205.add_inout(activation_scale_backing_inline26173);
                            params_t205.add_inout(payload_inline26172);
                            params_t205.add_input(projected_inline26169);
                            params_t205.add_scalar(output_workers_inline26175);
                            params_t205.add_scalar(active_rows_inline26083);
                            params_t205.launch_spec.set_block_num(output_workers_inline26175);
                            rt_submit_aiv_task(33, params_t205);
                            ret__tmp_v0_guarded_0_45 = activation_scale_backing_inline26173;
                            ret__tmp_v0_guarded_1_33 = payload_inline26172;
                        } else {
                            ret__tmp_v0_guarded_0_45 = activation_scale_backing_inline26173;
                            ret__tmp_v0_guarded_1_33 = payload_inline26172;
                        }
                        Tensor activation_scale_backing_inline26173__rv_v2 = ret__tmp_v0_guarded_0_45;
                        Tensor payload_inline26172__rv_v2 = ret__tmp_v0_guarded_1_33;
                        int64_t capacity_inline26182 = (int64_t)output_group_inline26087.shapes[0];

                        // Spmd proj_o_b_3: _o_proj_b
                        CoreTaskArgs params_t206;
                        params_t206.add_input(payload_inline26172__rv_v2);
                        params_t206.add_input(layer_wo_b_inline23473);
                        params_t206.add_input(activation_scale_backing_inline26173__rv_v2);
                        params_t206.add_input(layer_wo_b_scale_inline23474);
                        params_t206.add_inout(output_group_inline26087);
                        params_t206.add_scalar(capacity_inline26182);
                        params_t206.add_scalar((uint64_t)8192);
                        params_t206.launch_spec.set_block_num(20);
                        rt_submit_aic_task(34, params_t206);
                    }
                    TaskId hc_post_deps_inline24979[2]; for (int64_t __init_i = 0; __init_i < 2; ++__init_i) hc_post_deps_inline24979[__init_i] = TaskId::invalid();
                    uint32_t residual_flat_inline26216_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                    Tensor residual_flat_inline26216 = ext_x_hc.reshape(residual_flat_inline26216_shapes, 2);
                    uint32_t residual_mix_flat_inline26217_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 16};
                    Tensor residual_mix_flat_inline26217 = residual_mix_inline24945.reshape(residual_mix_flat_inline26217_shapes, 2);
                    uint32_t output_flat_inline26218_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                    Tensor output_flat_inline26218 = attention_hidden_inline23443.reshape(output_flat_inline26218_shapes, 2);
                    int64_t total_work_inline26219 = (LOCAL_T_DYN * 2);
                    int64_t full_work_inline26220 = ((total_work_inline26219 / 64) * 64);
                    int64_t tail_work_inline26221 = ((total_work_inline26219 - full_work_inline26220) * 4);
                    int64_t workers_inline26222 = std::min<int64_t>((total_work_inline26219 * 4), 64);
                    if ((0 < workers_inline26222)) {

                        // Spmd hc_post_spmd_4: hc_post
                        CoreTaskArgs params_t207;
                        params_t207.add_inout(output_flat_inline26218);
                        params_t207.add_input(attn_out_inline24978);
                        params_t207.add_input(residual_flat_inline26216);
                        params_t207.add_input(post_mix_inline24944__ssa_v1);
                        params_t207.add_input(residual_mix_flat_inline26217);
                        params_t207.add_scalar(full_work_inline26220);
                        params_t207.add_scalar(tail_work_inline26221);
                        params_t207.launch_spec.set_block_num(workers_inline26222);
                        TaskId params_t207_deps[2];
                        uint32_t params_t207_deps_count = 0;
                        if (hc_post_deps_inline24979[0].is_valid()) params_t207_deps[params_t207_deps_count++] = hc_post_deps_inline24979[0];
                        if (hc_post_deps_inline24979[1].is_valid()) params_t207_deps[params_t207_deps_count++] = hc_post_deps_inline24979[1];
                        params_t207.set_dependencies(params_t207_deps, params_t207_deps_count);
                        TaskOutputTensors task_207_outs = rt_submit_aiv_task(35, params_t207);
                    } else {

                        // Phase-fence barrier 43: dependency-only dummy task
                        CoreTaskArgs params_phase_fence_barrier_43;
                        TaskId params_phase_fence_barrier_43_deps[2];
                        uint32_t params_phase_fence_barrier_43_deps_count = 0;
                        if (hc_post_deps_inline24979[0].is_valid()) params_phase_fence_barrier_43_deps[params_phase_fence_barrier_43_deps_count++] = hc_post_deps_inline24979[0];
                        if (hc_post_deps_inline24979[1].is_valid()) params_phase_fence_barrier_43_deps[params_phase_fence_barrier_43_deps_count++] = hc_post_deps_inline24979[1];
                        params_phase_fence_barrier_43.set_dependencies(params_phase_fence_barrier_43_deps, params_phase_fence_barrier_43_deps_count);
                        TaskId spmd_empty_tid_35 = TaskId::invalid();
                        if (params_phase_fence_barrier_43_deps_count > 0) {
                            TaskOutputTensors phase_fence_barrier_43_outs = rt_submit_dummy_task(params_phase_fence_barrier_43);
                            spmd_empty_tid_35 = phase_fence_barrier_43_outs.task_id();
                        }
                    }
                    topk_indices_inline23445__rv_v3 = topk_indices_inline23445__rv_v2;
                    attention_pre_mix_inline23444__rv_v1 = attention_pre_mix_inline23444__ssa_v2;
                } else {
                    uint32_t post_mix_inline26270_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 4};
                    TensorCreateInfo post_mix_inline26270_ci(post_mix_inline26270_ci_shapes, 2, DataType::FLOAT32);
                    uint32_t residual_mix_inline26271_ci_shapes[3] = {static_cast<uint32_t>(LOCAL_T_DYN), 4, 4};
                    TensorCreateInfo residual_mix_inline26271_ci(residual_mix_inline26271_ci_shapes, 3, DataType::FLOAT32);
                    uint32_t x_norm_inline26272_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 5120};
                    TensorCreateInfo x_norm_inline26272_ci(x_norm_inline26272_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t qr_inline26274_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 1280};
                    TensorCreateInfo qr_inline26274_ci(qr_inline26274_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t query_inline26275_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 32768};
                    TensorCreateInfo query_inline26275_ci(query_inline26275_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t kv_inline26276_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 512};
                    TensorCreateInfo kv_inline26276_ci(kv_inline26276_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t kv_projection_inline26277_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 512};
                    TensorCreateInfo kv_projection_inline26277_ci(kv_projection_inline26277_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t attended_inline26281_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 32768};
                    TensorCreateInfo attended_inline26281_ci(attended_inline26281_ci_shapes, 2, DataType::BFLOAT16);
                    uint32_t attn_out_inline26291_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 5120};
                    TensorCreateInfo attn_out_inline26291_ci(attn_out_inline26291_ci_shapes, 2, DataType::BFLOAT16);
                    TaskOutputTensors alloc_83 = alloc_tensors(post_mix_inline26270_ci, residual_mix_inline26271_ci, x_norm_inline26272_ci, qr_inline26274_ci, query_inline26275_ci, kv_inline26276_ci, kv_projection_inline26277_ci, attended_inline26281_ci, attn_out_inline26291_ci);
                    const Tensor& post_mix_inline26270 = alloc_83.get_ref(0);
                    const Tensor& residual_mix_inline26271 = alloc_83.get_ref(1);
                    const Tensor& x_norm_inline26272 = alloc_83.get_ref(2);
                    const Tensor& qr_inline26274 = alloc_83.get_ref(3);
                    const Tensor& query_inline26275 = alloc_83.get_ref(4);
                    const Tensor& kv_inline26276 = alloc_83.get_ref(5);
                    const Tensor& kv_projection_inline26277 = alloc_83.get_ref(6);
                    const Tensor& attended_inline26281 = alloc_83.get_ref(7);
                    const Tensor& attn_out_inline26291 = alloc_83.get_ref(8);
                    uint32_t x_flat_inline26294_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                    Tensor x_flat_inline26294 = ext_x_hc.reshape(x_flat_inline26294_shapes, 2);
                    int64_t spmd_core_num_21 = ((LOCAL_T_DYN + 7) / 8);
                    Tensor ret__tmp_v0_guarded_0_46 = ext_engram_lookup;
                    TaskId ret__tmp_v0_guarded_1_34 = TaskId::invalid();
                    if ((0 < spmd_core_num_21)) {

                        // Spmd hc_collapse_norm_spmd_4: hc_collapse_norm
                        CoreTaskArgs params_t208;
                        params_t208.add_input(encoder_pre_mix__rv_v1);
                        params_t208.add_input(x_flat_inline26294);
                        params_t208.add_inout(x_norm_inline26272);
                        params_t208.add_input(layer_norm_weight_inline23461);
                        params_t208.add_scalar(LOCAL_T_DYN);
                        params_t208.launch_spec.set_block_num(spmd_core_num_21);
                        TaskOutputTensors task_208_outs = rt_submit_aiv_task(8, params_t208);
                        TaskId ret__tmp_v0_nonempty_1_36 = task_208_outs.task_id();
                        ret__tmp_v0_guarded_0_46 = x_norm_inline26272;
                        ret__tmp_v0_guarded_1_34 = ret__tmp_v0_nonempty_1_36;
                    } else {

                        // Phase-fence barrier 44: dependency-only dummy task
                        CoreTaskArgs params_phase_fence_barrier_44;
                        TaskOutputTensors phase_fence_barrier_44_outs = rt_submit_dummy_task(params_phase_fence_barrier_44);
                        TaskId spmd_empty_tid_36 = phase_fence_barrier_44_outs.task_id();
                        ret__tmp_v0_guarded_0_46 = x_norm_inline26272;
                        ret__tmp_v0_guarded_1_34 = spmd_empty_tid_36;
                    }
                    Tensor x_norm_inline26272__rv_v2 = ret__tmp_v0_guarded_0_46;
                    TaskId ready_inline26295 = ret__tmp_v0_guarded_1_34;
                    TaskId x_ready_inline26273 = ready_inline26295;
                    int64_t padded_rows_inline26342 = (((LOCAL_T_DYN + 79) / 80) * 80);
                    uint32_t payload_inline26343_ci_shapes[2] = {static_cast<uint32_t>(padded_rows_inline26342), 5120};
                    TensorCreateInfo payload_inline26343_ci(payload_inline26343_ci_shapes, 2, DataType::FP8E4M3FN);
                    TaskOutputTensors alloc_84 = alloc_tensors(payload_inline26343_ci);
                    const Tensor& payload_inline26343 = alloc_84.get_ref(0);
                    uint32_t scales_inline26344_ci_shapes[2] = {1, static_cast<uint32_t>((padded_rows_inline26342 * 160))};
                    TensorCreateInfo scales_inline26344_ci(scales_inline26344_ci_shapes, 2, DataType::FP8E8M0);
                    TaskOutputTensors alloc_85 = alloc_tensors(scales_inline26344_ci);
                    const Tensor& scales_inline26344 = alloc_85.get_ref(0);
                    int64_t quant_blocks_inline26345 = ((padded_rows_inline26342 / 16) * 20);
                    int32_t t__tmp_v1310 = rt_available_cluster_count();
                    int64_t workers_inline26346 = std::min<int64_t>(std::min<int64_t>(quant_blocks_inline26345, (static_cast<int64_t>(t__tmp_v1310) * 2)), 32);
                    Tensor ret__tmp_v0_guarded_0_47 = ext_engram_lookup;
                    Tensor ret__tmp_v0_guarded_1_35 = ext_engram_lookup;
                    if ((0 < workers_inline26346)) {

                        // Spmd attention_qkv_input_quant_pooled_spmd_4: attention_qkv_input_quant_pooled
                        CoreTaskArgs params_t209;
                        params_t209.add_inout(payload_inline26343);
                        params_t209.add_inout(scales_inline26344);
                        params_t209.add_input(x_norm_inline26272__rv_v2);
                        params_t209.add_scalar(quant_blocks_inline26345);
                        params_t209.add_scalar(workers_inline26346);
                        params_t209.add_scalar(attention_count_inline23434);
                        params_t209.launch_spec.set_block_num(workers_inline26346);
                        rt_submit_aiv_task(10, params_t209);
                        ret__tmp_v0_guarded_0_47 = payload_inline26343;
                        ret__tmp_v0_guarded_1_35 = scales_inline26344;
                    } else {
                        ret__tmp_v0_guarded_0_47 = payload_inline26343;
                        ret__tmp_v0_guarded_1_35 = scales_inline26344;
                    }
                    Tensor payload_inline26343__rv_v2 = ret__tmp_v0_guarded_0_47;
                    Tensor scales_inline26344__rv_v2 = ret__tmp_v0_guarded_1_35;
                    Tensor proj_payload_inline26279 = payload_inline26343__rv_v2;
                    Tensor proj_scales_inline26280 = scales_inline26344__rv_v2;
                    Tensor query_inline26389 = query_inline26275;
                    int64_t payload_rows_inline26390 = (int64_t)proj_payload_inline26279.shapes[0];
                    uint32_t qb_payload_inline26391_ci_shapes[2] = {static_cast<uint32_t>(payload_rows_inline26390), 1280};
                    TensorCreateInfo qb_payload_inline26391_ci(qb_payload_inline26391_ci_shapes, 2, DataType::FP8E4M3FN);
                    TaskOutputTensors alloc_86 = alloc_tensors(qb_payload_inline26391_ci);
                    const Tensor& qb_payload_inline26391 = alloc_86.get_ref(0);
                    uint32_t qb_scales_inline26392_ci_shapes[2] = {1, static_cast<uint32_t>((payload_rows_inline26390 * 40))};
                    TensorCreateInfo qb_scales_inline26392_ci(qb_scales_inline26392_ci_shapes, 2, DataType::FP8E8M0);
                    TaskOutputTensors alloc_87 = alloc_tensors(qb_scales_inline26392_ci);
                    const Tensor& qb_scales_inline26392 = alloc_87.get_ref(0);
                    Tensor query_inline26389__rv_v2 = query_inline26389;
                    for (int64_t group_start_inline26393 = 0; group_start_inline26393 < attention_count_inline23434; group_start_inline26393 += 80) {
                        uint32_t projected_inline26398_ci_shapes[2] = {80, 1280};
                        TensorCreateInfo projected_inline26398_ci(projected_inline26398_ci_shapes, 2, DataType::BFLOAT16);
                        TaskOutputTensors alloc_88 = alloc_tensors(projected_inline26398_ci);
                        const Tensor& projected_inline26398 = alloc_88.get_ref(0);
                        int64_t group_rows_inline26394 = std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) - group_start_inline26393), 80);
                        uint32_t group_payload_inline26395_offsets[2] = {static_cast<uint32_t>(group_start_inline26393), 0};
                        uint32_t group_payload_inline26395_shapes[2] = {(group_payload_inline26395_offsets[0] >= proj_payload_inline26279.shapes[0] ? 0u : std::min<uint32_t>(80, proj_payload_inline26279.shapes[0] - group_payload_inline26395_offsets[0])), (group_payload_inline26395_offsets[1] >= proj_payload_inline26279.shapes[1] ? 0u : std::min<uint32_t>(5120, proj_payload_inline26279.shapes[1] - group_payload_inline26395_offsets[1]))};
                        Tensor group_payload_inline26395 = proj_payload_inline26279.view(group_payload_inline26395_shapes, group_payload_inline26395_offsets);
                        int64_t scale_start_inline26396 = (group_start_inline26393 * 160);
                        uint32_t group_scales_inline26397_offsets[2] = {0, static_cast<uint32_t>(scale_start_inline26396)};
                        uint32_t group_scales_inline26397_shapes[2] = {(group_scales_inline26397_offsets[0] >= proj_scales_inline26280.shapes[0] ? 0u : std::min<uint32_t>(1, proj_scales_inline26280.shapes[0] - group_scales_inline26397_offsets[0])), (group_scales_inline26397_offsets[1] >= proj_scales_inline26280.shapes[1] ? 0u : std::min<uint32_t>(12800, proj_scales_inline26280.shapes[1] - group_scales_inline26397_offsets[1]))};
                        Tensor group_scales_inline26397 = proj_scales_inline26280.view(group_scales_inline26397_shapes, group_scales_inline26397_offsets);

                        // Spmd proj_q_a_4: _proj_q_a
                        CoreTaskArgs params_t210;
                        params_t210.add_input(group_payload_inline26395);
                        params_t210.add_input(layer_wq_a_inline23462);
                        params_t210.add_input(group_scales_inline26397);
                        params_t210.add_input(layer_wq_a_scale_inline23463);
                        params_t210.add_inout(projected_inline26398);
                        params_t210.add_scalar(group_rows_inline26394);
                        params_t210.launch_spec.set_block_num(20);
                        rt_submit_aic_task(11, params_t210);
                        uint32_t qb_payload_group_inline26399_offsets[2] = {static_cast<uint32_t>(group_start_inline26393), 0};
                        uint32_t qb_payload_group_inline26399_shapes[2] = {(qb_payload_group_inline26399_offsets[0] >= qb_payload_inline26391.shapes[0] ? 0u : std::min<uint32_t>(80, qb_payload_inline26391.shapes[0] - qb_payload_group_inline26399_offsets[0])), (qb_payload_group_inline26399_offsets[1] >= qb_payload_inline26391.shapes[1] ? 0u : std::min<uint32_t>(1280, qb_payload_inline26391.shapes[1] - qb_payload_group_inline26399_offsets[1]))};
                        Tensor qb_payload_group_inline26399 = qb_payload_inline26391.view(qb_payload_group_inline26399_shapes, qb_payload_group_inline26399_offsets);
                        int64_t qb_scale_start_inline26400 = (group_start_inline26393 * 40);
                        uint32_t qb_scales_group_inline26401_offsets[2] = {0, static_cast<uint32_t>(qb_scale_start_inline26400)};
                        uint32_t qb_scales_group_inline26401_shapes[2] = {(qb_scales_group_inline26401_offsets[0] >= qb_scales_inline26392.shapes[0] ? 0u : std::min<uint32_t>(1, qb_scales_inline26392.shapes[0] - qb_scales_group_inline26401_offsets[0])), (qb_scales_group_inline26401_offsets[1] >= qb_scales_inline26392.shapes[1] ? 0u : std::min<uint32_t>(3200, qb_scales_inline26392.shapes[1] - qb_scales_group_inline26401_offsets[1]))};
                        Tensor qb_scales_group_inline26401 = qb_scales_inline26392.view(qb_scales_group_inline26401_shapes, qb_scales_group_inline26401_offsets);
                        for (int64_t q_mt_inline26402 = 0; q_mt_inline26402 < 5; q_mt_inline26402 += 1) {
                            int64_t q_local_inline26403 = (q_mt_inline26402 * 16);
                            int64_t q_start_inline26404 = (group_start_inline26393 + std::min<int64_t>(q_local_inline26403, group_rows_inline26394));
                            int64_t q_rows_inline26405 = std::max<int64_t>(std::min<int64_t>((group_rows_inline26394 - q_local_inline26403), 16), 0);
                            uint32_t projected_tile_inline26406_offsets[2] = {static_cast<uint32_t>(q_local_inline26403), 0};
                            uint32_t projected_tile_inline26406_shapes[2] = {(projected_tile_inline26406_offsets[0] >= projected_inline26398.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(q_rows_inline26405), projected_inline26398.shapes[0] - projected_tile_inline26406_offsets[0])), (projected_tile_inline26406_offsets[1] >= projected_inline26398.shapes[1] ? 0u : std::min<uint32_t>(1280, projected_inline26398.shapes[1] - projected_tile_inline26406_offsets[1]))};
                            Tensor projected_tile_inline26406 = projected_inline26398.view(projected_tile_inline26406_shapes, projected_tile_inline26406_offsets);
                            uint32_t latent_tile_inline26407_offsets[2] = {static_cast<uint32_t>(q_start_inline26404), 0};
                            uint32_t latent_tile_inline26407_shapes[2] = {(latent_tile_inline26407_offsets[0] >= qr_inline26274.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(q_rows_inline26405), qr_inline26274.shapes[0] - latent_tile_inline26407_offsets[0])), (latent_tile_inline26407_offsets[1] >= qr_inline26274.shapes[1] ? 0u : std::min<uint32_t>(1280, qr_inline26274.shapes[1] - latent_tile_inline26407_offsets[1]))};
                            Tensor latent_tile_inline26407 = qr_inline26274.view(latent_tile_inline26407_shapes, latent_tile_inline26407_offsets);
                            uint32_t qb_payload_tile_inline26408_offsets[2] = {static_cast<uint32_t>(q_local_inline26403), 0};
                            uint32_t qb_payload_tile_inline26408_shapes[2] = {(qb_payload_tile_inline26408_offsets[0] >= qb_payload_group_inline26399.shapes[0] ? 0u : std::min<uint32_t>(16, qb_payload_group_inline26399.shapes[0] - qb_payload_tile_inline26408_offsets[0])), (qb_payload_tile_inline26408_offsets[1] >= qb_payload_group_inline26399.shapes[1] ? 0u : std::min<uint32_t>(1280, qb_payload_group_inline26399.shapes[1] - qb_payload_tile_inline26408_offsets[1]))};
                            Tensor qb_payload_tile_inline26408 = qb_payload_group_inline26399.view(qb_payload_tile_inline26408_shapes, qb_payload_tile_inline26408_offsets);
                            int64_t qb_scale_offset_inline26409 = (q_local_inline26403 * 40);
                            uint32_t qb_scales_tile_inline26410_offsets[2] = {0, static_cast<uint32_t>(qb_scale_offset_inline26409)};
                            uint32_t qb_scales_tile_inline26410_shapes[2] = {(qb_scales_tile_inline26410_offsets[0] >= qb_scales_group_inline26401.shapes[0] ? 0u : std::min<uint32_t>(1, qb_scales_group_inline26401.shapes[0] - qb_scales_tile_inline26410_offsets[0])), (qb_scales_tile_inline26410_offsets[1] >= qb_scales_group_inline26401.shapes[1] ? 0u : std::min<uint32_t>(640, qb_scales_group_inline26401.shapes[1] - qb_scales_tile_inline26410_offsets[1]))};
                            Tensor qb_scales_tile_inline26410 = qb_scales_group_inline26401.view(qb_scales_tile_inline26410_shapes, qb_scales_tile_inline26410_offsets);
                            uint32_t weight_row_inline26415_shapes[2] = {1, 1280};
                            Tensor weight_row_inline26415 = layer_q_norm_inline23464.reshape(weight_row_inline26415_shapes, 2);

                            // Spmd attention_q_norm_quant_spmd_4: attention_q_norm_quant_4
                            CoreTaskArgs params_t211;
                            params_t211.add_input(projected_tile_inline26406);
                            params_t211.add_input(weight_row_inline26415);
                            params_t211.add_inout(latent_tile_inline26407);
                            params_t211.add_inout(qb_payload_tile_inline26408);
                            params_t211.add_inout(qb_scales_tile_inline26410);
                            params_t211.add_scalar(q_rows_inline26405);
                            params_t211.launch_spec.set_block_num(5);
                            rt_submit_aiv_task(118, params_t211);
                        }
                        int32_t group_tokens_inline26411 = static_cast<int32_t>(group_rows_inline26394);
                        int32_t t__tmp_v1319 = rt_available_cluster_count();
                        int64_t qb_workers_v1_inline26413 = std::min<int64_t>(static_cast<int64_t>(t__tmp_v1319), 32);
                        int64_t qb_workers_v2_inline26414 = std::max<int64_t>(qb_workers_v1_inline26413, 1);

                        // Spmd proj_q_b_4: _proj_q_b
                        CoreTaskArgs params_t212;
                        params_t212.add_input(qb_payload_group_inline26399);
                        params_t212.add_input(layer_wq_b_inline23465);
                        params_t212.add_input(qb_scales_group_inline26401);
                        params_t212.add_input(layer_wq_b_scale_inline23466);
                        params_t212.add_inout(query_inline26389__rv_v2);
                        params_t212.add_scalar(group_rows_inline26394);
                        params_t212.add_scalar((uint64_t)32768);
                        params_t212.add_scalar(group_start_inline26393);
                        params_t212.launch_spec.set_block_num(qb_workers_v2_inline26414);
                        rt_submit_aic_task(13, params_t212);
                        int32_t t__tmp_v1320 = rt_available_cluster_count();
                        int64_t rope_workers_inline26475 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v1320) * 2), 32);
                        Tensor query_inline26389__rv_v5 = ext_engram_lookup;
                        if ((0 < rope_workers_inline26475)) {

                            // Spmd attention_q_rope_spmd_4: attention_q_rope
                            CoreTaskArgs params_t213;
                            params_t213.add_inout(query_inline26389__rv_v2);
                            params_t213.add_input(compressed_token_rope_cos_inline23437);
                            params_t213.add_input(compressed_token_rope_sin_inline23438);
                            params_t213.add_scalar(rope_workers_inline26475);
                            params_t213.add_scalar(group_start_inline26393);
                            params_t213.add_scalar(group_tokens_inline26411);
                            params_t213.launch_spec.set_block_num(rope_workers_inline26475);
                            rt_submit_aiv_task(14, params_t213);
                            query_inline26389__rv_v5 = query_inline26389__rv_v2;
                        } else {
                            query_inline26389__rv_v5 = query_inline26389__rv_v2;
                        }
                        query_inline26389__rv_v2 = query_inline26389__rv_v5;
                    }
                    Tensor kv_projection_inline26509 = kv_projection_inline26277;
                    Tensor kv_inline26276__rv_v2 = kv_inline26276;
                    for (int64_t group_start_inline26510 = 0; group_start_inline26510 < attention_count_inline23434; group_start_inline26510 += 80) {
                        int64_t group_rows_inline26511 = std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) - group_start_inline26510), 80);
                        uint32_t group_payload_inline26512_offsets[2] = {static_cast<uint32_t>(group_start_inline26510), 0};
                        uint32_t group_payload_inline26512_shapes[2] = {(group_payload_inline26512_offsets[0] >= proj_payload_inline26279.shapes[0] ? 0u : std::min<uint32_t>(80, proj_payload_inline26279.shapes[0] - group_payload_inline26512_offsets[0])), (group_payload_inline26512_offsets[1] >= proj_payload_inline26279.shapes[1] ? 0u : std::min<uint32_t>(5120, proj_payload_inline26279.shapes[1] - group_payload_inline26512_offsets[1]))};
                        Tensor group_payload_inline26512 = proj_payload_inline26279.view(group_payload_inline26512_shapes, group_payload_inline26512_offsets);
                        int64_t scale_start_inline26513 = (group_start_inline26510 * 160);
                        uint32_t group_scales_inline26514_offsets[2] = {0, static_cast<uint32_t>(scale_start_inline26513)};
                        uint32_t group_scales_inline26514_shapes[2] = {(group_scales_inline26514_offsets[0] >= proj_scales_inline26280.shapes[0] ? 0u : std::min<uint32_t>(1, proj_scales_inline26280.shapes[0] - group_scales_inline26514_offsets[0])), (group_scales_inline26514_offsets[1] >= proj_scales_inline26280.shapes[1] ? 0u : std::min<uint32_t>(12800, proj_scales_inline26280.shapes[1] - group_scales_inline26514_offsets[1]))};
                        Tensor group_scales_inline26514 = proj_scales_inline26280.view(group_scales_inline26514_shapes, group_scales_inline26514_offsets);

                        // Spmd proj_kv_4: _proj_kv
                        CoreTaskArgs params_t214;
                        params_t214.add_input(group_payload_inline26512);
                        params_t214.add_input(layer_wkv_inline23467);
                        params_t214.add_input(group_scales_inline26514);
                        params_t214.add_input(layer_wkv_scale_inline23468);
                        params_t214.add_inout(kv_projection_inline26509);
                        params_t214.add_scalar(group_rows_inline26511);
                        params_t214.add_scalar(group_start_inline26510);
                        params_t214.launch_spec.set_block_num(8);
                        params_t214.launch_spec.set_require_sync_start(true);
                        rt_submit_aic_task(15, params_t214);
                        uint32_t projection_group_inline26515_offsets[2] = {static_cast<uint32_t>(group_start_inline26510), 0};
                        uint32_t projection_group_inline26515_shapes[2] = {(projection_group_inline26515_offsets[0] >= kv_projection_inline26509.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline26511), kv_projection_inline26509.shapes[0] - projection_group_inline26515_offsets[0])), (projection_group_inline26515_offsets[1] >= kv_projection_inline26509.shapes[1] ? 0u : std::min<uint32_t>(512, kv_projection_inline26509.shapes[1] - projection_group_inline26515_offsets[1]))};
                        Tensor projection_group_inline26515 = kv_projection_inline26509.view(projection_group_inline26515_shapes, projection_group_inline26515_offsets);
                        uint32_t normalized_group_inline26516_offsets[2] = {static_cast<uint32_t>(group_start_inline26510), 0};
                        uint32_t normalized_group_inline26516_shapes[2] = {(normalized_group_inline26516_offsets[0] >= kv_inline26276__rv_v2.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline26511), kv_inline26276__rv_v2.shapes[0] - normalized_group_inline26516_offsets[0])), (normalized_group_inline26516_offsets[1] >= kv_inline26276__rv_v2.shapes[1] ? 0u : std::min<uint32_t>(512, kv_inline26276__rv_v2.shapes[1] - normalized_group_inline26516_offsets[1]))};
                        Tensor normalized_group_inline26516 = kv_inline26276__rv_v2.view(normalized_group_inline26516_shapes, normalized_group_inline26516_offsets);
                        int32_t group_tokens_inline26517 = static_cast<int32_t>(group_rows_inline26511);
                        int64_t spmd_core_num_22 = ((static_cast<int64_t>(group_tokens_inline26517) + 7) / 8);
                        if ((0 < spmd_core_num_22)) {

                            // Spmd norm_kv_spmd_4: norm_kv
                            CoreTaskArgs params_t215;
                            params_t215.add_input(projection_group_inline26515);
                            params_t215.add_input(layer_kv_norm_inline23469);
                            params_t215.add_inout(normalized_group_inline26516);
                            params_t215.add_scalar(group_tokens_inline26517);
                            params_t215.launch_spec.set_block_num(spmd_core_num_22);
                            rt_submit_aiv_task(16, params_t215);
                        } else {
                        }
                        int32_t t__tmp_v1331 = rt_available_cluster_count();
                        int64_t rope_workers_inline26528 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v1331) * 2), 5);
                        Tensor kv_inline26276__rv_v4 = ext_engram_lookup;
                        if ((0 < rope_workers_inline26528)) {

                            // Spmd rope_kv_spmd_4: rope_kv
                            CoreTaskArgs params_t216;
                            params_t216.add_inout(kv_inline26276__rv_v2);
                            params_t216.add_input(compressed_token_rope_cos_inline23437);
                            params_t216.add_input(compressed_token_rope_sin_inline23438);
                            params_t216.add_scalar(rope_workers_inline26528);
                            params_t216.add_scalar(group_start_inline26510);
                            params_t216.add_scalar(group_tokens_inline26517);
                            params_t216.launch_spec.set_block_num(rope_workers_inline26528);
                            rt_submit_aiv_task(17, params_t216);
                            kv_inline26276__rv_v4 = kv_inline26276__rv_v2;
                        } else {
                            kv_inline26276__rv_v4 = kv_inline26276__rv_v2;
                        }
                        kv_inline26276__rv_v2 = kv_inline26276__rv_v4;
                    }
                    int64_t workers_inline26562 = std::min<int64_t>(static_cast<int64_t>(attention_count_inline23434), 12);
                    int64_t step_inline26563 = std::max<int64_t>(workers_inline26562, 1);
                    if ((0 < workers_inline26562)) {

                        // Spmd pub_win_spmd_4: pub_win_4
                        CoreTaskArgs params_t217;
                        params_t217.add_inout(layer_window_cache_inline23453);
                        params_t217.add_inout(layer_window_cache_scale_inline23454);
                        params_t217.add_input(ext_window_slots);
                        params_t217.add_input(kv_inline26276__rv_v2);
                        params_t217.add_scalar(attention_count_inline23434);
                        params_t217.add_scalar(step_inline26563);
                        params_t217.launch_spec.set_block_num(workers_inline26562);
                        rt_submit_aiv_task(119, params_t217);
                    } else {
                    }
                    if ((static_cast<int64_t>(attention_count_inline23434) <= 12)) {
                        uint32_t selected_a_inline26588_ci_shapes[3] = {24, 640, 512};
                        TensorCreateInfo selected_a_inline26588_ci(selected_a_inline26588_ci_shapes, 3, DataType::BFLOAT16);
                        uint32_t selected_b_inline26589_ci_shapes[3] = {24, 640, 512};
                        TensorCreateInfo selected_b_inline26589_ci(selected_b_inline26589_ci_shapes, 3, DataType::BFLOAT16);
                        uint32_t combined_a_inline26590_ci_shapes[2] = {24, 640};
                        TensorCreateInfo combined_a_inline26590_ci(combined_a_inline26590_ci_shapes, 2, DataType::INT32);
                        uint32_t combined_b_inline26591_ci_shapes[2] = {24, 640};
                        TensorCreateInfo combined_b_inline26591_ci(combined_b_inline26591_ci_shapes, 2, DataType::INT32);
                        TaskOutputTensors alloc_89 = alloc_tensors(selected_a_inline26588_ci, selected_b_inline26589_ci, combined_a_inline26590_ci, combined_b_inline26591_ci);
                        const Tensor& selected_a_inline26588 = alloc_89.get_ref(0);
                        const Tensor& selected_b_inline26589 = alloc_89.get_ref(1);
                        const Tensor& combined_a_inline26590 = alloc_89.get_ref(2);
                        const Tensor& combined_b_inline26591 = alloc_89.get_ref(3);
                        TaskId done_a_inline26592 = x_ready_inline26273;
                        TaskId done_b_inline26593 = x_ready_inline26273;
                        TaskId done_a_inline26592__rv_v2 = done_a_inline26592;
                        TaskId done_b_inline26593__rv_v2 = done_b_inline26593;
                        Tensor combined_b_inline26591__rv_v2 = combined_b_inline26591;
                        for (int64_t start_inline26594 = 0; start_inline26594 < attention_count_inline23434; start_inline26594 += 48) {
                            int64_t width_inline26595 = std::min<int64_t>((((static_cast<int64_t>(attention_count_inline23434) - start_inline26594) + 1) / 2), 24);
                            int64_t active_a_inline26596 = width_inline26595;
                            uint32_t wi_a_inline26597_offsets[2] = {static_cast<uint32_t>(start_inline26594), 0};
                            uint32_t wi_a_inline26597_shapes[2] = {(wi_a_inline26597_offsets[0] >= ext_window_indices.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_a_inline26596), ext_window_indices.shapes[0] - wi_a_inline26597_offsets[0])), (wi_a_inline26597_offsets[1] >= ext_window_indices.shapes[1] ? 0u : std::min<uint32_t>(128, ext_window_indices.shapes[1] - wi_a_inline26597_offsets[1]))};
                            Tensor wi_a_inline26597 = ext_window_indices.view(wi_a_inline26597_shapes, wi_a_inline26597_offsets);
                            uint32_t ci_a_inline26598_offsets[2] = {static_cast<uint32_t>(start_inline26594), 0};
                            uint32_t ci_a_inline26598_shapes[2] = {(ci_a_inline26598_offsets[0] >= topk_indices_inline23445__rv_v5.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_a_inline26596), topk_indices_inline23445__rv_v5.shapes[0] - ci_a_inline26598_offsets[0])), (ci_a_inline26598_offsets[1] >= topk_indices_inline23445__rv_v5.shapes[1] ? 0u : std::min<uint32_t>(512, topk_indices_inline23445__rv_v5.shapes[1] - ci_a_inline26598_offsets[1]))};
                            Tensor ci_a_inline26598 = topk_indices_inline23445__rv_v5.view(ci_a_inline26598_shapes, ci_a_inline26598_offsets);
                            uint32_t query_a_inline26599_offsets[2] = {static_cast<uint32_t>(start_inline26594), 0};
                            uint32_t query_a_inline26599_shapes[2] = {(query_a_inline26599_offsets[0] >= query_inline26275.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_a_inline26596), query_inline26275.shapes[0] - query_a_inline26599_offsets[0])), (query_a_inline26599_offsets[1] >= query_inline26275.shapes[1] ? 0u : std::min<uint32_t>(32768, query_inline26275.shapes[1] - query_a_inline26599_offsets[1]))};
                            Tensor query_a_inline26599 = query_inline26275.view(query_a_inline26599_shapes, query_a_inline26599_offsets);
                            uint32_t output_a_inline26600_offsets[2] = {static_cast<uint32_t>(start_inline26594), 0};
                            uint32_t output_a_inline26600_shapes[2] = {(output_a_inline26600_offsets[0] >= attended_inline26281.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_a_inline26596), attended_inline26281.shapes[0] - output_a_inline26600_offsets[0])), (output_a_inline26600_offsets[1] >= attended_inline26281.shapes[1] ? 0u : std::min<uint32_t>(32768, attended_inline26281.shapes[1] - output_a_inline26600_offsets[1]))};
                            Tensor output_a_inline26600 = attended_inline26281.view(output_a_inline26600_shapes, output_a_inline26600_offsets);
                            int64_t t__tmp_v1334 = (int64_t)layer_window_cache_inline23453.shapes[0];
                            int64_t window_rows_inline26611 = (t__tmp_v1334 * 128);
                            int64_t t__tmp_v1335 = (int64_t)layer_compressed_cache_inline23456.shapes[0];
                            int64_t compressed_rows_inline26612 = (t__tmp_v1335 * 128);
                            uint32_t w_inline26613_shapes[2] = {static_cast<uint32_t>(window_rows_inline26611), 512};
                            Tensor w_inline26613 = layer_window_cache_inline23453.reshape(w_inline26613_shapes, 2);
                            uint32_t ws_inline26614_shapes[2] = {static_cast<uint32_t>(window_rows_inline26611), 16};
                            Tensor ws_inline26614 = layer_window_cache_scale_inline23454.reshape(ws_inline26614_shapes, 2);
                            uint32_t c_inline26615_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline26612), 256};
                            Tensor c_inline26615 = layer_compressed_cache_inline23456.reshape(c_inline26615_shapes, 2);
                            uint32_t cs_inline26616_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline26612), 32};
                            Tensor cs_inline26616 = layer_compressed_cache_scale_inline23457.reshape(cs_inline26616_shapes, 2);
                            int64_t t__tmp_v1336 = 24;
                            int64_t selected_rows_inline26617 = (t__tmp_v1336 * 640);
                            uint32_t dst_inline26618_shapes[2] = {static_cast<uint32_t>(selected_rows_inline26617), 512};
                            Tensor dst_inline26618 = selected_a_inline26588.reshape(dst_inline26618_shapes, 2);
                            int64_t workers_inline26619 = std::max<int64_t>(std::min<int64_t>((static_cast<int64_t>(active_a_inline26596) * 20), 32), 1);

                            // Spmd c1a_gather_spmd_3: c1a_gather_0
                            CoreTaskArgs params_t218;
                            params_t218.add_input(wi_a_inline26597);
                            params_t218.add_input(w_inline26613);
                            params_t218.add_input(ws_inline26614);
                            params_t218.add_input(ci_a_inline26598);
                            params_t218.add_input(c_inline26615);
                            params_t218.add_input(cs_inline26616);
                            params_t218.add_inout(dst_inline26618);
                            params_t218.add_inout(combined_a_inline26590);
                            params_t218.add_scalar(active_a_inline26596);
                            params_t218.add_scalar(workers_inline26619);
                            params_t218.launch_spec.set_block_num(workers_inline26619);
                            TaskId params_t218_deps[1];
                            uint32_t params_t218_deps_count = 0;
                            if (done_a_inline26592__rv_v2.is_valid()) params_t218_deps[params_t218_deps_count++] = done_a_inline26592__rv_v2;
                            params_t218.set_dependencies(params_t218_deps, params_t218_deps_count);
                            TaskOutputTensors task_218_outs = rt_submit_aiv_task(57, params_t218);
                            TaskId gathered_inline26620 = task_218_outs.task_id();
                            TaskId gathered_a_inline26602 = gathered_inline26620;
                            int64_t tokens_inline26658 = (int64_t)query_a_inline26599.shapes[0];
                            int64_t head_rows_inline26659 = (tokens_inline26658 * 64);
                            uint32_t query_flat_inline26660_shapes[2] = {static_cast<uint32_t>(head_rows_inline26659), 512};
                            Tensor query_flat_inline26660 = query_a_inline26599.reshape(query_flat_inline26660_shapes, 2);
                            uint32_t output_flat_inline26661_shapes[2] = {static_cast<uint32_t>(head_rows_inline26659), 512};
                            Tensor output_flat_inline26661 = output_a_inline26600.reshape(output_flat_inline26661_shapes, 2);
                            uint32_t key_flat_inline26662_shapes[2] = {15360, 512};
                            Tensor key_flat_inline26662 = selected_a_inline26588.reshape(key_flat_inline26662_shapes, 2);
                            int64_t attend_tasks_inline26663 = (static_cast<int64_t>(active_a_inline26596) * 2);
                            int64_t workers_inline26664 = std::min<int64_t>(attend_tasks_inline26663, 16);
                            int32_t t__tmp_v1341 = rt_available_cluster_count();
                            int64_t workers_inline26664__ssa_v3 = std::min<int64_t>(workers_inline26664, static_cast<int64_t>(t__tmp_v1341));
                            TaskId ret__tmp_v0_guarded_0_48 = TaskId::invalid();
                            if ((0 < workers_inline26664__ssa_v3)) {

                                // Group c1a_sparse_attn: MixedKernels (AIC + AIV lanes)
                                CoreTaskArgs params_t219;
                                params_t219.add_inout(output_flat_inline26661);
                                params_t219.add_input(query_flat_inline26660);
                                params_t219.add_input(key_flat_inline26662);
                                params_t219.add_input(combined_a_inline26590);
                                params_t219.add_input(layer_sink_inline23470);
                                params_t219.add_scalar(attend_tasks_inline26663);
                                params_t219.add_scalar(workers_inline26664__ssa_v3);
                                MixedKernels mixed_219 = {58, 59, INVALID_KERNEL_ID};
                                params_t219.launch_spec.set_block_num(workers_inline26664__ssa_v3);
                                params_t219.launch_spec.set_require_sync_start(true);
                                TaskId params_t219_deps[1];
                                uint32_t params_t219_deps_count = 0;
                                params_t219_deps[params_t219_deps_count++] = gathered_inline26620;
                                params_t219.set_dependencies(params_t219_deps, params_t219_deps_count);
                                TaskOutputTensors task_219_outs = rt_submit_task(mixed_219, params_t219);
                                TaskId ret__tmp_v0_nonempty_0_54 = task_219_outs.task_id();
                                ret__tmp_v0_guarded_0_48 = ret__tmp_v0_nonempty_0_54;
                            } else {

                                // Phase-fence barrier 45: dependency-only dummy task
                                CoreTaskArgs params_phase_fence_barrier_45;
                                TaskId params_phase_fence_barrier_45_deps[1];
                                uint32_t params_phase_fence_barrier_45_deps_count = 0;
                                params_phase_fence_barrier_45_deps[params_phase_fence_barrier_45_deps_count++] = gathered_inline26620;
                                params_phase_fence_barrier_45.set_dependencies(params_phase_fence_barrier_45_deps, params_phase_fence_barrier_45_deps_count);
                                TaskId spmd_empty_tid_37 = TaskId::invalid();
                                if (params_phase_fence_barrier_45_deps_count > 0) {
                                    TaskOutputTensors phase_fence_barrier_45_outs = rt_submit_dummy_task(params_phase_fence_barrier_45);
                                    spmd_empty_tid_37 = phase_fence_barrier_45_outs.task_id();
                                }
                                ret__tmp_v0_guarded_0_48 = spmd_empty_tid_37;
                            }
                            TaskId attend_tid_inline26665 = ret__tmp_v0_guarded_0_48;
                            TaskId done_a_inline26592__ssa_v3 = attend_tid_inline26665;
                            TaskId done_b_inline26593__phi_v4 = TaskId::invalid();
                            Tensor combined_b_inline26591__rv_v1 = ext_engram_lookup;
                            if (((start_inline26594 + width_inline26595) < static_cast<int64_t>(attention_count_inline23434))) {
                                int64_t active_b_inline26603 = std::min<int64_t>(((static_cast<int64_t>(attention_count_inline23434) - start_inline26594) - width_inline26595), 24);
                                uint32_t wi_b_inline26604_offsets[2] = {static_cast<uint32_t>((start_inline26594 + width_inline26595)), 0};
                                uint32_t wi_b_inline26604_shapes[2] = {(wi_b_inline26604_offsets[0] >= ext_window_indices.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_b_inline26603), ext_window_indices.shapes[0] - wi_b_inline26604_offsets[0])), (wi_b_inline26604_offsets[1] >= ext_window_indices.shapes[1] ? 0u : std::min<uint32_t>(128, ext_window_indices.shapes[1] - wi_b_inline26604_offsets[1]))};
                                Tensor wi_b_inline26604 = ext_window_indices.view(wi_b_inline26604_shapes, wi_b_inline26604_offsets);
                                uint32_t ci_b_inline26605_offsets[2] = {static_cast<uint32_t>((start_inline26594 + width_inline26595)), 0};
                                uint32_t ci_b_inline26605_shapes[2] = {(ci_b_inline26605_offsets[0] >= topk_indices_inline23445__rv_v5.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_b_inline26603), topk_indices_inline23445__rv_v5.shapes[0] - ci_b_inline26605_offsets[0])), (ci_b_inline26605_offsets[1] >= topk_indices_inline23445__rv_v5.shapes[1] ? 0u : std::min<uint32_t>(512, topk_indices_inline23445__rv_v5.shapes[1] - ci_b_inline26605_offsets[1]))};
                                Tensor ci_b_inline26605 = topk_indices_inline23445__rv_v5.view(ci_b_inline26605_shapes, ci_b_inline26605_offsets);
                                uint32_t query_b_inline26606_offsets[2] = {static_cast<uint32_t>((start_inline26594 + width_inline26595)), 0};
                                uint32_t query_b_inline26606_shapes[2] = {(query_b_inline26606_offsets[0] >= query_inline26275.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_b_inline26603), query_inline26275.shapes[0] - query_b_inline26606_offsets[0])), (query_b_inline26606_offsets[1] >= query_inline26275.shapes[1] ? 0u : std::min<uint32_t>(32768, query_inline26275.shapes[1] - query_b_inline26606_offsets[1]))};
                                Tensor query_b_inline26606 = query_inline26275.view(query_b_inline26606_shapes, query_b_inline26606_offsets);
                                uint32_t output_b_inline26607_offsets[2] = {static_cast<uint32_t>((start_inline26594 + width_inline26595)), 0};
                                uint32_t output_b_inline26607_shapes[2] = {(output_b_inline26607_offsets[0] >= attended_inline26281.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(active_b_inline26603), attended_inline26281.shapes[0] - output_b_inline26607_offsets[0])), (output_b_inline26607_offsets[1] >= attended_inline26281.shapes[1] ? 0u : std::min<uint32_t>(32768, attended_inline26281.shapes[1] - output_b_inline26607_offsets[1]))};
                                Tensor output_b_inline26607 = attended_inline26281.view(output_b_inline26607_shapes, output_b_inline26607_offsets);
                                int64_t t__tmp_v1360 = (int64_t)layer_window_cache_inline23453.shapes[0];
                                int64_t window_rows_inline26699 = (t__tmp_v1360 * 128);
                                int64_t t__tmp_v1361 = (int64_t)layer_compressed_cache_inline23456.shapes[0];
                                int64_t compressed_rows_inline26700 = (t__tmp_v1361 * 128);
                                uint32_t w_inline26701_shapes[2] = {static_cast<uint32_t>(window_rows_inline26699), 512};
                                Tensor w_inline26701 = layer_window_cache_inline23453.reshape(w_inline26701_shapes, 2);
                                uint32_t ws_inline26702_shapes[2] = {static_cast<uint32_t>(window_rows_inline26699), 16};
                                Tensor ws_inline26702 = layer_window_cache_scale_inline23454.reshape(ws_inline26702_shapes, 2);
                                uint32_t c_inline26703_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline26700), 256};
                                Tensor c_inline26703 = layer_compressed_cache_inline23456.reshape(c_inline26703_shapes, 2);
                                uint32_t cs_inline26704_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline26700), 32};
                                Tensor cs_inline26704 = layer_compressed_cache_scale_inline23457.reshape(cs_inline26704_shapes, 2);
                                int64_t t__tmp_v1362 = 24;
                                int64_t selected_rows_inline26705 = (t__tmp_v1362 * 640);
                                uint32_t dst_inline26706_shapes[2] = {static_cast<uint32_t>(selected_rows_inline26705), 512};
                                Tensor dst_inline26706 = selected_b_inline26589.reshape(dst_inline26706_shapes, 2);
                                int64_t workers_inline26707 = std::max<int64_t>(std::min<int64_t>((static_cast<int64_t>(active_b_inline26603) * 20), 32), 1);

                                // Spmd c1a_gather_spmd_4: c1a_gather_0
                                CoreTaskArgs params_t220;
                                params_t220.add_input(wi_b_inline26604);
                                params_t220.add_input(w_inline26701);
                                params_t220.add_input(ws_inline26702);
                                params_t220.add_input(ci_b_inline26605);
                                params_t220.add_input(c_inline26703);
                                params_t220.add_input(cs_inline26704);
                                params_t220.add_inout(dst_inline26706);
                                params_t220.add_inout(combined_b_inline26591__rv_v2);
                                params_t220.add_scalar(active_b_inline26603);
                                params_t220.add_scalar(workers_inline26707);
                                params_t220.launch_spec.set_block_num(workers_inline26707);
                                TaskId params_t220_deps[1];
                                uint32_t params_t220_deps_count = 0;
                                if (done_b_inline26593__rv_v2.is_valid()) params_t220_deps[params_t220_deps_count++] = done_b_inline26593__rv_v2;
                                params_t220.set_dependencies(params_t220_deps, params_t220_deps_count);
                                TaskOutputTensors task_220_outs = rt_submit_aiv_task(57, params_t220);
                                TaskId gathered_inline26708 = task_220_outs.task_id();
                                TaskId gathered_b_inline26609 = gathered_inline26708;
                                int64_t tokens_inline26746 = (int64_t)query_b_inline26606.shapes[0];
                                int64_t head_rows_inline26747 = (tokens_inline26746 * 64);
                                uint32_t query_flat_inline26748_shapes[2] = {static_cast<uint32_t>(head_rows_inline26747), 512};
                                Tensor query_flat_inline26748 = query_b_inline26606.reshape(query_flat_inline26748_shapes, 2);
                                uint32_t output_flat_inline26749_shapes[2] = {static_cast<uint32_t>(head_rows_inline26747), 512};
                                Tensor output_flat_inline26749 = output_b_inline26607.reshape(output_flat_inline26749_shapes, 2);
                                uint32_t key_flat_inline26750_shapes[2] = {15360, 512};
                                Tensor key_flat_inline26750 = selected_b_inline26589.reshape(key_flat_inline26750_shapes, 2);
                                int64_t attend_tasks_inline26751 = (static_cast<int64_t>(active_b_inline26603) * 2);
                                int64_t workers_inline26752 = std::min<int64_t>(attend_tasks_inline26751, 16);
                                int32_t t__tmp_v1367 = rt_available_cluster_count();
                                int64_t workers_inline26752__ssa_v3 = std::min<int64_t>(workers_inline26752, static_cast<int64_t>(t__tmp_v1367));
                                TaskId ret__tmp_v0_guarded_0_49 = TaskId::invalid();
                                if ((0 < workers_inline26752__ssa_v3)) {

                                    // Group c1a_sparse_attn: MixedKernels (AIC + AIV lanes)
                                    CoreTaskArgs params_t221;
                                    params_t221.add_inout(output_flat_inline26749);
                                    params_t221.add_input(query_flat_inline26748);
                                    params_t221.add_input(key_flat_inline26750);
                                    params_t221.add_input(combined_b_inline26591__rv_v2);
                                    params_t221.add_input(layer_sink_inline23470);
                                    params_t221.add_scalar(attend_tasks_inline26751);
                                    params_t221.add_scalar(workers_inline26752__ssa_v3);
                                    MixedKernels mixed_221 = {58, 59, INVALID_KERNEL_ID};
                                    params_t221.launch_spec.set_block_num(workers_inline26752__ssa_v3);
                                    params_t221.launch_spec.set_require_sync_start(true);
                                    TaskId params_t221_deps[1];
                                    uint32_t params_t221_deps_count = 0;
                                    params_t221_deps[params_t221_deps_count++] = gathered_inline26708;
                                    params_t221.set_dependencies(params_t221_deps, params_t221_deps_count);
                                    TaskOutputTensors task_221_outs = rt_submit_task(mixed_221, params_t221);
                                    TaskId ret__tmp_v0_nonempty_0_55 = task_221_outs.task_id();
                                    ret__tmp_v0_guarded_0_49 = ret__tmp_v0_nonempty_0_55;
                                } else {

                                    // Phase-fence barrier 46: dependency-only dummy task
                                    CoreTaskArgs params_phase_fence_barrier_46;
                                    TaskId params_phase_fence_barrier_46_deps[1];
                                    uint32_t params_phase_fence_barrier_46_deps_count = 0;
                                    params_phase_fence_barrier_46_deps[params_phase_fence_barrier_46_deps_count++] = gathered_inline26708;
                                    params_phase_fence_barrier_46.set_dependencies(params_phase_fence_barrier_46_deps, params_phase_fence_barrier_46_deps_count);
                                    TaskId spmd_empty_tid_38 = TaskId::invalid();
                                    if (params_phase_fence_barrier_46_deps_count > 0) {
                                        TaskOutputTensors phase_fence_barrier_46_outs = rt_submit_dummy_task(params_phase_fence_barrier_46);
                                        spmd_empty_tid_38 = phase_fence_barrier_46_outs.task_id();
                                    }
                                    ret__tmp_v0_guarded_0_49 = spmd_empty_tid_38;
                                }
                                TaskId attend_tid_inline26753 = ret__tmp_v0_guarded_0_49;
                                TaskId done_b_inline26593__ssa_v3 = attend_tid_inline26753;
                                done_b_inline26593__phi_v4 = done_b_inline26593__ssa_v3;
                                combined_b_inline26591__rv_v1 = combined_b_inline26591__rv_v2;
                            } else {
                                done_b_inline26593__phi_v4 = done_b_inline26593__rv_v2;
                                combined_b_inline26591__rv_v1 = combined_b_inline26591__rv_v2;
                            }
                            done_a_inline26592__rv_v2 = done_a_inline26592__ssa_v3;
                            done_b_inline26593__rv_v2 = done_b_inline26593__phi_v4;
                            combined_b_inline26591__rv_v2 = combined_b_inline26591__rv_v1;
                        }

                        // Phase-fence barrier 47: dependency-only dummy task
                        CoreTaskArgs params_phase_fence_barrier_47;
                        TaskId params_phase_fence_barrier_47_deps[2];
                        uint32_t params_phase_fence_barrier_47_deps_count = 0;
                        if (done_a_inline26592__rv_v2.is_valid()) params_phase_fence_barrier_47_deps[params_phase_fence_barrier_47_deps_count++] = done_a_inline26592__rv_v2;
                        if (done_b_inline26593__rv_v2.is_valid()) params_phase_fence_barrier_47_deps[params_phase_fence_barrier_47_deps_count++] = done_b_inline26593__rv_v2;
                        params_phase_fence_barrier_47.set_dependencies(params_phase_fence_barrier_47_deps, params_phase_fence_barrier_47_deps_count);
                        TaskId complete_inline26610 = TaskId::invalid();
                        if (params_phase_fence_barrier_47_deps_count > 0) {
                            TaskOutputTensors phase_fence_barrier_47_outs = rt_submit_dummy_task(params_phase_fence_barrier_47);
                            complete_inline26610 = phase_fence_barrier_47_outs.task_id();
                        }
                    } else {
                        uint32_t selected_inline26282_ci_shapes[3] = {static_cast<uint32_t>(LOCAL_T_DYN), 640, 512};
                        TensorCreateInfo selected_inline26282_ci(selected_inline26282_ci_shapes, 3, DataType::BFLOAT16);
                        uint32_t indices_inline26283_ci_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 640};
                        TensorCreateInfo indices_inline26283_ci(indices_inline26283_ci_shapes, 2, DataType::INT32);
                        TaskOutputTensors alloc_90 = alloc_tensors(selected_inline26282_ci, indices_inline26283_ci);
                        const Tensor& selected_inline26282 = alloc_90.get_ref(0);
                        const Tensor& indices_inline26283 = alloc_90.get_ref(1);
                        int64_t t__tmp_v1386 = (int64_t)layer_window_cache_inline23453.shapes[0];
                        int64_t window_rows_inline26787 = (t__tmp_v1386 * 128);
                        int64_t t__tmp_v1387 = (int64_t)layer_compressed_cache_inline23456.shapes[0];
                        int64_t compressed_rows_inline26788 = (t__tmp_v1387 * 128);
                        uint32_t w_inline26789_shapes[2] = {static_cast<uint32_t>(window_rows_inline26787), 512};
                        Tensor w_inline26789 = layer_window_cache_inline23453.reshape(w_inline26789_shapes, 2);
                        uint32_t ws_inline26790_shapes[2] = {static_cast<uint32_t>(window_rows_inline26787), 16};
                        Tensor ws_inline26790 = layer_window_cache_scale_inline23454.reshape(ws_inline26790_shapes, 2);
                        uint32_t c_inline26791_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline26788), 256};
                        Tensor c_inline26791 = layer_compressed_cache_inline23456.reshape(c_inline26791_shapes, 2);
                        uint32_t cs_inline26792_shapes[2] = {static_cast<uint32_t>(compressed_rows_inline26788), 32};
                        Tensor cs_inline26792 = layer_compressed_cache_scale_inline23457.reshape(cs_inline26792_shapes, 2);
                        int64_t selected_rows_inline26793 = (LOCAL_T_DYN * 640);
                        uint32_t dst_inline26794_shapes[2] = {static_cast<uint32_t>(selected_rows_inline26793), 512};
                        Tensor dst_inline26794 = selected_inline26282.reshape(dst_inline26794_shapes, 2);
                        int64_t workers_inline26795 = std::max<int64_t>(std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) * 20), 64), 1);

                        // Spmd c1a_gather_spmd_5: c1a_gather
                        CoreTaskArgs params_t222;
                        params_t222.add_input(ext_window_indices);
                        params_t222.add_input(w_inline26789);
                        params_t222.add_input(ws_inline26790);
                        params_t222.add_input(topk_indices_inline23445__rv_v5);
                        params_t222.add_input(c_inline26791);
                        params_t222.add_input(cs_inline26792);
                        params_t222.add_inout(dst_inline26794);
                        params_t222.add_inout(indices_inline26283);
                        params_t222.add_scalar(attention_count_inline23434);
                        params_t222.add_scalar(workers_inline26795);
                        params_t222.launch_spec.set_block_num(workers_inline26795);
                        TaskId params_t222_deps[1];
                        uint32_t params_t222_deps_count = 0;
                        if (ret__tmp_v0_guarded_1_34.is_valid()) params_t222_deps[params_t222_deps_count++] = ret__tmp_v0_guarded_1_34;
                        params_t222.set_dependencies(params_t222_deps, params_t222_deps_count);
                        TaskOutputTensors task_222_outs = rt_submit_aiv_task(60, params_t222);
                        TaskId gathered_inline26796 = task_222_outs.task_id();
                        TaskId gather_tid_inline26287 = gathered_inline26796;
                        int64_t head_rows_inline26835 = (LOCAL_T_DYN * 64);
                        int64_t cache_rows_inline26836 = (LOCAL_T_DYN * 640);
                        uint32_t qflat_inline26837_shapes[2] = {static_cast<uint32_t>(head_rows_inline26835), 512};
                        Tensor qflat_inline26837 = query_inline26275.reshape(qflat_inline26837_shapes, 2);
                        uint32_t kflat_inline26838_shapes[2] = {static_cast<uint32_t>(cache_rows_inline26836), 512};
                        Tensor kflat_inline26838 = selected_inline26282.reshape(kflat_inline26838_shapes, 2);
                        uint32_t oflat_inline26839_shapes[2] = {static_cast<uint32_t>(head_rows_inline26835), 512};
                        Tensor oflat_inline26839 = attended_inline26281.reshape(oflat_inline26839_shapes, 2);
                        int64_t workers_inline26840 = std::max<int64_t>(std::min<int64_t>((static_cast<int64_t>(attention_count_inline23434) * 2), 32), 1);

                        // Group sparse_online_cube: MixedKernels (AIC + AIV lanes)
                        CoreTaskArgs params_t223;
                        params_t223.add_inout(oflat_inline26839);
                        params_t223.add_input(qflat_inline26837);
                        params_t223.add_input(kflat_inline26838);
                        params_t223.add_input(indices_inline26283);
                        params_t223.add_input(layer_sink_inline23470);
                        params_t223.add_scalar(attention_count_inline23434);
                        params_t223.add_scalar(workers_inline26840);
                        MixedKernels mixed_223 = {61, 62, INVALID_KERNEL_ID};
                        params_t223.launch_spec.set_block_num(workers_inline26840);
                        TaskId params_t223_deps[1];
                        uint32_t params_t223_deps_count = 0;
                        params_t223_deps[params_t223_deps_count++] = gathered_inline26796;
                        params_t223.set_dependencies(params_t223_deps, params_t223_deps_count);
                        TaskOutputTensors task_223_outs = rt_submit_task(mixed_223, params_t223);
                    }
                    int64_t t_linear_inline26885 = (((LOCAL_T_DYN + 15) / 16) * 16);
                    uint32_t x_flat_inline26886_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                    Tensor x_flat_inline26886 = ext_x_hc.reshape(x_flat_inline26886_shapes, 2);
                    uint32_t inv_rms_inline26887_ci_shapes[2] = {static_cast<uint32_t>(t_linear_inline26885), 1};
                    TensorCreateInfo inv_rms_inline26887_ci(inv_rms_inline26887_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_91 = alloc_tensors(inv_rms_inline26887_ci);
                    const Tensor& inv_rms_inline26887 = alloc_91.get_ref(0);
                    int64_t spmd_core_num_23 = ((LOCAL_T_DYN + 7) / 8);
                    Tensor inv_rms_inline26887__ssa_v1 = ext_engram_lookup;
                    if ((0 < spmd_core_num_23)) {

                        // Spmd mhc_rms_spmd_5: mhc_rms
                        CoreTaskArgs params_t224;
                        params_t224.add_input(x_flat_inline26886);
                        params_t224.add_inout(inv_rms_inline26887);
                        params_t224.add_scalar(LOCAL_T_DYN);
                        params_t224.launch_spec.set_block_num(spmd_core_num_23);
                        rt_submit_aiv_task(28, params_t224);
                        inv_rms_inline26887__ssa_v1 = inv_rms_inline26887;
                    } else {
                        inv_rms_inline26887__ssa_v1 = inv_rms_inline26887;
                    }
                    uint32_t mixes_partials_inline26896_ci_shapes[2] = {static_cast<uint32_t>((t_linear_inline26885 * 4)), 32};
                    TensorCreateInfo mixes_partials_inline26896_ci(mixes_partials_inline26896_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_92 = alloc_tensors(mixes_partials_inline26896_ci);
                    const Tensor& mixes_partials_inline26896 = alloc_92.get_ref(0);
                    int64_t tasks_inline26897 = (((LOCAL_T_DYN + 79) / 80) * 4);
                    int32_t t__tmp_v1409 = rt_available_cluster_count();
                    int64_t workers_inline26898 = std::min<int64_t>(tasks_inline26897, static_cast<int64_t>(t__tmp_v1409));
                    if ((0 < workers_inline26898)) {

                        // Spmd mhc_linear_5: _hc_linear
                        CoreTaskArgs params_t225;
                        params_t225.add_input(x_flat_inline26886);
                        params_t225.add_input(layer_hc_fn_inline23458);
                        params_t225.add_inout(mixes_partials_inline26896);
                        params_t225.add_scalar(LOCAL_T_DYN);
                        params_t225.add_scalar(t_linear_inline26885);
                        params_t225.launch_spec.set_block_num(workers_inline26898);
                        rt_submit_aic_task(29, params_t225);
                    } else {
                    }
                    Tensor inv_rms_inline26289 = inv_rms_inline26887__ssa_v1;
                    Tensor partials_inline26290 = mixes_partials_inline26896;
                    int64_t linear_tokens_inline26908 = (int64_t)inv_rms_inline26289.shapes[0];
                    uint32_t base_row_inline26909_shapes[2] = {1, 24};
                    Tensor base_row_inline26909 = layer_hc_base_inline23460.reshape(base_row_inline26909_shapes, 2);
                    uint32_t residual_flat_inline26910_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 16};
                    Tensor residual_flat_inline26910 = residual_mix_inline26271.reshape(residual_flat_inline26910_shapes, 2);
                    uint32_t indices_scale0_inline26911[1] = {0};
                    float scale0_inline26911 = get_tensor_data<float>(layer_hc_scale_inline23459, 1, indices_scale0_inline26911);
                    uint32_t indices_scale1_inline26912[1] = {1};
                    float scale1_inline26912 = get_tensor_data<float>(layer_hc_scale_inline23459, 1, indices_scale1_inline26912);
                    uint32_t indices_scale2_inline26913[1] = {2};
                    float scale2_inline26913 = get_tensor_data<float>(layer_hc_scale_inline23459, 1, indices_scale2_inline26913);
                    int64_t blocks_inline26914 = ((LOCAL_T_DYN + 1) / 2);
                    int64_t workers_inline26915 = std::min<int64_t>(blocks_inline26914, 12);
                    Tensor ret__tmp_v0_guarded_0_50 = ext_engram_lookup;
                    Tensor ret__tmp_v0_guarded_1_36 = ext_engram_lookup;
                    if ((0 < workers_inline26915)) {

                        // Spmd mhc_sinkhorn_spmd_5: mhc_sinkhorn
                        CoreTaskArgs params_t226;
                        params_t226.add_input(partials_inline26290);
                        params_t226.add_input(inv_rms_inline26289);
                        params_t226.add_input(base_row_inline26909);
                        params_t226.add_inout(attention_pre_mix_inline23444__rv_v3);
                        params_t226.add_inout(post_mix_inline26270);
                        params_t226.add_inout(residual_flat_inline26910);
                        params_t226.add_scalar(LOCAL_T_DYN);
                        params_t226.add_scalar(blocks_inline26914);
                        params_t226.add_scalar(workers_inline26915);
                        params_t226.add_scalar(linear_tokens_inline26908);
                        params_t226.add_scalar(to_u64(scale0_inline26911));
                        params_t226.add_scalar(to_u64(scale1_inline26912));
                        params_t226.add_scalar(to_u64(scale2_inline26913));
                        params_t226.launch_spec.set_block_num(workers_inline26915);
                        TaskOutputTensors task_226_outs = rt_submit_aiv_task(30, params_t226);
                        ret__tmp_v0_guarded_0_50 = attention_pre_mix_inline23444__rv_v3;
                        ret__tmp_v0_guarded_1_36 = post_mix_inline26270;
                    } else {

                        // Phase-fence barrier 48: dependency-only dummy task
                        CoreTaskArgs params_phase_fence_barrier_48;
                        TaskOutputTensors phase_fence_barrier_48_outs = rt_submit_dummy_task(params_phase_fence_barrier_48);
                        TaskId spmd_empty_tid_39 = phase_fence_barrier_48_outs.task_id();
                        ret__tmp_v0_guarded_0_50 = attention_pre_mix_inline23444__rv_v3;
                        ret__tmp_v0_guarded_1_36 = post_mix_inline26270;
                    }
                    Tensor attention_pre_mix_inline23444__ssa_v3 = ret__tmp_v0_guarded_0_50;
                    Tensor post_mix_inline26270__ssa_v1 = ret__tmp_v0_guarded_1_36;
                    for (int64_t group_start_inline27020 = 0; group_start_inline27020 < LOCAL_T_DYN; group_start_inline27020 += 80) {
                        uint32_t payload_all_inline27030_ci_shapes[2] = {640, 4096};
                        TensorCreateInfo payload_all_inline27030_ci(payload_all_inline27030_ci_shapes, 2, DataType::FP8E4M3FN);
                        uint32_t scales_all_inline27031_ci_shapes[2] = {8, 10240};
                        TensorCreateInfo scales_all_inline27031_ci(scales_all_inline27031_ci_shapes, 2, DataType::FP8E8M0);
                        uint32_t projected_inline27108_ci_shapes[2] = {80, 8192};
                        TensorCreateInfo projected_inline27108_ci(projected_inline27108_ci_shapes, 2, DataType::BFLOAT16);
                        uint32_t payload_inline27111_ci_shapes[2] = {80, 8192};
                        TensorCreateInfo payload_inline27111_ci(payload_inline27111_ci_shapes, 2, DataType::FP8E4M3FN);
                        uint32_t activation_scale_backing_inline27112_ci_shapes[2] = {1, 20480};
                        TensorCreateInfo activation_scale_backing_inline27112_ci(activation_scale_backing_inline27112_ci_shapes, 2, DataType::FP8E8M0);
                        TaskOutputTensors alloc_93 = alloc_tensors(payload_all_inline27030_ci, scales_all_inline27031_ci, projected_inline27108_ci, payload_inline27111_ci, activation_scale_backing_inline27112_ci);
                        const Tensor& payload_all_inline27030 = alloc_93.get_ref(0);
                        const Tensor& scales_all_inline27031 = alloc_93.get_ref(1);
                        const Tensor& projected_inline27108 = alloc_93.get_ref(2);
                        const Tensor& payload_inline27111 = alloc_93.get_ref(3);
                        const Tensor& activation_scale_backing_inline27112 = alloc_93.get_ref(4);
                        int64_t group_rows_inline27021 = std::min<int64_t>((LOCAL_T_DYN - group_start_inline27020), 80);
                        int64_t active_rows_inline27022 = std::max<int64_t>(std::min<int64_t>(group_rows_inline27021, (static_cast<int64_t>(attention_count_inline23434) - group_start_inline27020)), 0);
                        uint32_t attended_group_inline27023_offsets[2] = {static_cast<uint32_t>(group_start_inline27020), 0};
                        uint32_t attended_group_inline27023_shapes[2] = {(attended_group_inline27023_offsets[0] >= attended_inline26281.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline27021), attended_inline26281.shapes[0] - attended_group_inline27023_offsets[0])), (attended_group_inline27023_offsets[1] >= attended_inline26281.shapes[1] ? 0u : std::min<uint32_t>(32768, attended_inline26281.shapes[1] - attended_group_inline27023_offsets[1]))};
                        Tensor attended_group_inline27023 = attended_inline26281.view(attended_group_inline27023_shapes, attended_group_inline27023_offsets);
                        uint32_t cosine_group_inline27024_offsets[2] = {static_cast<uint32_t>(group_start_inline27020), 0};
                        uint32_t cosine_group_inline27024_shapes[2] = {(cosine_group_inline27024_offsets[0] >= compressed_token_rope_cos_inline23437.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline27021), compressed_token_rope_cos_inline23437.shapes[0] - cosine_group_inline27024_offsets[0])), (cosine_group_inline27024_offsets[1] >= compressed_token_rope_cos_inline23437.shapes[1] ? 0u : std::min<uint32_t>(32, compressed_token_rope_cos_inline23437.shapes[1] - cosine_group_inline27024_offsets[1]))};
                        Tensor cosine_group_inline27024 = compressed_token_rope_cos_inline23437.view(cosine_group_inline27024_shapes, cosine_group_inline27024_offsets);
                        uint32_t sine_group_inline27025_offsets[2] = {static_cast<uint32_t>(group_start_inline27020), 0};
                        uint32_t sine_group_inline27025_shapes[2] = {(sine_group_inline27025_offsets[0] >= compressed_token_rope_sin_inline23438.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline27021), compressed_token_rope_sin_inline23438.shapes[0] - sine_group_inline27025_offsets[0])), (sine_group_inline27025_offsets[1] >= compressed_token_rope_sin_inline23438.shapes[1] ? 0u : std::min<uint32_t>(32, compressed_token_rope_sin_inline23438.shapes[1] - sine_group_inline27025_offsets[1]))};
                        Tensor sine_group_inline27025 = compressed_token_rope_sin_inline23438.view(sine_group_inline27025_shapes, sine_group_inline27025_offsets);
                        uint32_t output_group_inline27026_offsets[2] = {static_cast<uint32_t>(group_start_inline27020), 0};
                        uint32_t output_group_inline27026_shapes[2] = {(output_group_inline27026_offsets[0] >= attn_out_inline26291.shapes[0] ? 0u : std::min<uint32_t>(static_cast<uint32_t>(group_rows_inline27021), attn_out_inline26291.shapes[0] - output_group_inline27026_offsets[0])), (output_group_inline27026_offsets[1] >= attn_out_inline26291.shapes[1] ? 0u : std::min<uint32_t>(5120, attn_out_inline26291.shapes[1] - output_group_inline27026_offsets[1]))};
                        Tensor output_group_inline27026 = attn_out_inline26291.view(output_group_inline27026_shapes, output_group_inline27026_offsets);
                        int64_t t__tmp_v1410 = (int64_t)attended_group_inline27023.shapes[0];
                        int64_t last_input_row_inline27029 = std::max<int64_t>((t__tmp_v1410 - 1), 0);
                        int32_t t__tmp_v1411 = rt_available_cluster_count();
                        int64_t quant_workers_inline27034 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v1411) * 2), 64);
                        Tensor ret__tmp_v0_guarded_0_51 = ext_engram_lookup;
                        Tensor ret__tmp_v0_guarded_1_37 = ext_engram_lookup;
                        if ((0 < quant_workers_inline27034)) {

                            // Spmd rope_quant_o_pooled_spmd_4: rope_quant_o_pooled
                            CoreTaskArgs params_t227;
                            params_t227.add_inout(payload_all_inline27030);
                            params_t227.add_inout(scales_all_inline27031);
                            params_t227.add_input(cosine_group_inline27024);
                            params_t227.add_input(sine_group_inline27025);
                            params_t227.add_input(attended_group_inline27023);
                            params_t227.add_scalar(quant_workers_inline27034);
                            params_t227.add_scalar(active_rows_inline27022);
                            params_t227.add_scalar(last_input_row_inline27029);
                            params_t227.launch_spec.set_block_num(quant_workers_inline27034);
                            rt_submit_aiv_task(31, params_t227);
                            ret__tmp_v0_guarded_0_51 = payload_all_inline27030;
                            ret__tmp_v0_guarded_1_37 = scales_all_inline27031;
                        } else {
                            ret__tmp_v0_guarded_0_51 = payload_all_inline27030;
                            ret__tmp_v0_guarded_1_37 = scales_all_inline27031;
                        }
                        Tensor payload_all_inline27030__rv_v2 = ret__tmp_v0_guarded_0_51;
                        Tensor scales_all_inline27031__rv_v2 = ret__tmp_v0_guarded_1_37;
                        int32_t t__tmp_v1414 = rt_available_cluster_count();
                        int64_t oa_workers_inline27109 = std::min<int64_t>(static_cast<int64_t>(t__tmp_v1414), 32);
                        int64_t oa_workers_v1_inline27110 = std::max<int64_t>(oa_workers_inline27109, 1);

                        // Spmd proj_o_a_4: _o_proj_a
                        CoreTaskArgs params_t228;
                        params_t228.add_input(payload_all_inline27030__rv_v2);
                        params_t228.add_input(layer_wo_a_inline23471);
                        params_t228.add_input(scales_all_inline27031__rv_v2);
                        params_t228.add_input(layer_wo_a_scale_inline23472);
                        params_t228.add_inout(projected_inline27108);
                        params_t228.add_scalar((uint64_t)8192);
                        params_t228.launch_spec.set_block_num(oa_workers_v1_inline27110);
                        rt_submit_aic_task(32, params_t228);
                        int32_t t__tmp_v1415 = rt_available_cluster_count();
                        int64_t output_workers_inline27114 = std::min<int64_t>((static_cast<int64_t>(t__tmp_v1415) * 2), 64);
                        Tensor ret__tmp_v0_guarded_0_52 = ext_engram_lookup;
                        Tensor ret__tmp_v0_guarded_1_38 = ext_engram_lookup;
                        if ((0 < output_workers_inline27114)) {

                            // Spmd quant_o_a_spmd_4: quant_o_a
                            CoreTaskArgs params_t229;
                            params_t229.add_inout(activation_scale_backing_inline27112);
                            params_t229.add_inout(payload_inline27111);
                            params_t229.add_input(projected_inline27108);
                            params_t229.add_scalar(output_workers_inline27114);
                            params_t229.add_scalar(active_rows_inline27022);
                            params_t229.launch_spec.set_block_num(output_workers_inline27114);
                            rt_submit_aiv_task(33, params_t229);
                            ret__tmp_v0_guarded_0_52 = activation_scale_backing_inline27112;
                            ret__tmp_v0_guarded_1_38 = payload_inline27111;
                        } else {
                            ret__tmp_v0_guarded_0_52 = activation_scale_backing_inline27112;
                            ret__tmp_v0_guarded_1_38 = payload_inline27111;
                        }
                        Tensor activation_scale_backing_inline27112__rv_v2 = ret__tmp_v0_guarded_0_52;
                        Tensor payload_inline27111__rv_v2 = ret__tmp_v0_guarded_1_38;
                        int64_t capacity_inline27121 = (int64_t)output_group_inline27026.shapes[0];

                        // Spmd proj_o_b_4: _o_proj_b
                        CoreTaskArgs params_t230;
                        params_t230.add_input(payload_inline27111__rv_v2);
                        params_t230.add_input(layer_wo_b_inline23473);
                        params_t230.add_input(activation_scale_backing_inline27112__rv_v2);
                        params_t230.add_input(layer_wo_b_scale_inline23474);
                        params_t230.add_inout(output_group_inline27026);
                        params_t230.add_scalar(capacity_inline27121);
                        params_t230.add_scalar((uint64_t)8192);
                        params_t230.launch_spec.set_block_num(20);
                        rt_submit_aic_task(34, params_t230);
                    }
                    TaskId hc_post_deps_inline26292[2]; for (int64_t __init_i = 0; __init_i < 2; ++__init_i) hc_post_deps_inline26292[__init_i] = TaskId::invalid();
                    uint32_t residual_flat_inline27155_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                    Tensor residual_flat_inline27155 = ext_x_hc.reshape(residual_flat_inline27155_shapes, 2);
                    uint32_t residual_mix_flat_inline27156_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 16};
                    Tensor residual_mix_flat_inline27156 = residual_mix_inline26271.reshape(residual_mix_flat_inline27156_shapes, 2);
                    uint32_t output_flat_inline27157_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
                    Tensor output_flat_inline27157 = attention_hidden_inline23443.reshape(output_flat_inline27157_shapes, 2);
                    int64_t total_work_inline27158 = (LOCAL_T_DYN * 2);
                    int64_t full_work_inline27159 = ((total_work_inline27158 / 64) * 64);
                    int64_t tail_work_inline27160 = ((total_work_inline27158 - full_work_inline27159) * 4);
                    int64_t workers_inline27161 = std::min<int64_t>((total_work_inline27158 * 4), 64);
                    if ((0 < workers_inline27161)) {

                        // Spmd hc_post_spmd_5: hc_post
                        CoreTaskArgs params_t231;
                        params_t231.add_inout(output_flat_inline27157);
                        params_t231.add_input(attn_out_inline26291);
                        params_t231.add_input(residual_flat_inline27155);
                        params_t231.add_input(post_mix_inline26270__ssa_v1);
                        params_t231.add_input(residual_mix_flat_inline27156);
                        params_t231.add_scalar(full_work_inline27159);
                        params_t231.add_scalar(tail_work_inline27160);
                        params_t231.launch_spec.set_block_num(workers_inline27161);
                        TaskId params_t231_deps[2];
                        uint32_t params_t231_deps_count = 0;
                        if (hc_post_deps_inline26292[0].is_valid()) params_t231_deps[params_t231_deps_count++] = hc_post_deps_inline26292[0];
                        if (hc_post_deps_inline26292[1].is_valid()) params_t231_deps[params_t231_deps_count++] = hc_post_deps_inline26292[1];
                        params_t231.set_dependencies(params_t231_deps, params_t231_deps_count);
                        TaskOutputTensors task_231_outs = rt_submit_aiv_task(35, params_t231);
                    } else {

                        // Phase-fence barrier 49: dependency-only dummy task
                        CoreTaskArgs params_phase_fence_barrier_49;
                        TaskId params_phase_fence_barrier_49_deps[2];
                        uint32_t params_phase_fence_barrier_49_deps_count = 0;
                        if (hc_post_deps_inline26292[0].is_valid()) params_phase_fence_barrier_49_deps[params_phase_fence_barrier_49_deps_count++] = hc_post_deps_inline26292[0];
                        if (hc_post_deps_inline26292[1].is_valid()) params_phase_fence_barrier_49_deps[params_phase_fence_barrier_49_deps_count++] = hc_post_deps_inline26292[1];
                        params_phase_fence_barrier_49.set_dependencies(params_phase_fence_barrier_49_deps, params_phase_fence_barrier_49_deps_count);
                        TaskId spmd_empty_tid_40 = TaskId::invalid();
                        if (params_phase_fence_barrier_49_deps_count > 0) {
                            TaskOutputTensors phase_fence_barrier_49_outs = rt_submit_dummy_task(params_phase_fence_barrier_49);
                            spmd_empty_tid_40 = phase_fence_barrier_49_outs.task_id();
                        }
                    }
                    topk_indices_inline23445__rv_v3 = topk_indices_inline23445__rv_v5;
                    attention_pre_mix_inline23444__rv_v1 = attention_pre_mix_inline23444__ssa_v3;
                }
                attention_pre_mix_inline23444__rv_v2 = attention_pre_mix_inline23444__rv_v1;
                topk_indices_inline23445__rv_v4 = topk_indices_inline23445__rv_v3;
                pool_block_ids_inline23448__rv_v2 = pool_block_ids_inline23448__rv_v3;
            }
            int64_t ffn_routed_base_inline23490 = (layer_inline23452 * 64);
            uint32_t ffn_hc_fn_inline23491_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 24)), 0};
            uint32_t ffn_hc_fn_inline23491_shapes[2] = {(ffn_hc_fn_inline23491_offsets[0] >= ext_hc_ffn_fn.shapes[0] ? 0u : std::min<uint32_t>(24, ext_hc_ffn_fn.shapes[0] - ffn_hc_fn_inline23491_offsets[0])), (ffn_hc_fn_inline23491_offsets[1] >= ext_hc_ffn_fn.shapes[1] ? 0u : std::min<uint32_t>(20480, ext_hc_ffn_fn.shapes[1] - ffn_hc_fn_inline23491_offsets[1]))};
            Tensor ffn_hc_fn_inline23491 = ext_hc_ffn_fn.view(ffn_hc_fn_inline23491_shapes, ffn_hc_fn_inline23491_offsets);
            uint32_t ffn_hc_scale_inline23492_offsets[1] = {static_cast<uint32_t>((layer_inline23452 * 3))};
            uint32_t ffn_hc_scale_inline23492_shapes[1] = {(ffn_hc_scale_inline23492_offsets[0] >= ext_hc_ffn_scale.shapes[0] ? 0u : std::min<uint32_t>(3, ext_hc_ffn_scale.shapes[0] - ffn_hc_scale_inline23492_offsets[0]))};
            Tensor ffn_hc_scale_inline23492 = ext_hc_ffn_scale.view(ffn_hc_scale_inline23492_shapes, ffn_hc_scale_inline23492_offsets);
            uint32_t ffn_hc_base_inline23493_offsets[1] = {static_cast<uint32_t>((layer_inline23452 * 24))};
            uint32_t ffn_hc_base_inline23493_shapes[1] = {(ffn_hc_base_inline23493_offsets[0] >= ext_hc_ffn_base.shapes[0] ? 0u : std::min<uint32_t>(24, ext_hc_ffn_base.shapes[0] - ffn_hc_base_inline23493_offsets[0]))};
            Tensor ffn_hc_base_inline23493 = ext_hc_ffn_base.view(ffn_hc_base_inline23493_shapes, ffn_hc_base_inline23493_offsets);
            uint32_t layer_ffn_norm_inline23494_offsets[1] = {static_cast<uint32_t>((layer_inline23452 * 5120))};
            uint32_t layer_ffn_norm_inline23494_shapes[1] = {(layer_ffn_norm_inline23494_offsets[0] >= ext_ffn_norm_weight.shapes[0] ? 0u : std::min<uint32_t>(5120, ext_ffn_norm_weight.shapes[0] - layer_ffn_norm_inline23494_offsets[0]))};
            Tensor layer_ffn_norm_inline23494 = ext_ffn_norm_weight.view(layer_ffn_norm_inline23494_shapes, layer_ffn_norm_inline23494_offsets);
            uint32_t ffn_gate_weight_inline23495_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 128)), 0};
            uint32_t ffn_gate_weight_inline23495_shapes[2] = {(ffn_gate_weight_inline23495_offsets[0] >= ext_gate_weight.shapes[0] ? 0u : std::min<uint32_t>(128, ext_gate_weight.shapes[0] - ffn_gate_weight_inline23495_offsets[0])), (ffn_gate_weight_inline23495_offsets[1] >= ext_gate_weight.shapes[1] ? 0u : std::min<uint32_t>(5120, ext_gate_weight.shapes[1] - ffn_gate_weight_inline23495_offsets[1]))};
            Tensor ffn_gate_weight_inline23495 = ext_gate_weight.view(ffn_gate_weight_inline23495_shapes, ffn_gate_weight_inline23495_offsets);
            uint32_t ffn_correction_bias_inline23496_offsets[1] = {static_cast<uint32_t>((layer_inline23452 * 128))};
            uint32_t ffn_correction_bias_inline23496_shapes[1] = {(ffn_correction_bias_inline23496_offsets[0] >= ext_correction_bias.shapes[0] ? 0u : std::min<uint32_t>(128, ext_correction_bias.shapes[0] - ffn_correction_bias_inline23496_offsets[0]))};
            Tensor ffn_correction_bias_inline23496 = ext_correction_bias.view(ffn_correction_bias_inline23496_shapes, ffn_correction_bias_inline23496_offsets);
            uint32_t ffn_routed_w1_inline23497_offsets[3] = {static_cast<uint32_t>(ffn_routed_base_inline23490), 0, 0};
            uint32_t ffn_routed_w1_inline23497_shapes[3] = {(ffn_routed_w1_inline23497_offsets[0] >= ext_routed_w1.shapes[0] ? 0u : std::min<uint32_t>(64, ext_routed_w1.shapes[0] - ffn_routed_w1_inline23497_offsets[0])), (ffn_routed_w1_inline23497_offsets[1] >= ext_routed_w1.shapes[1] ? 0u : std::min<uint32_t>(46080, ext_routed_w1.shapes[1] - ffn_routed_w1_inline23497_offsets[1])), (ffn_routed_w1_inline23497_offsets[2] >= ext_routed_w1.shapes[2] ? 0u : std::min<uint32_t>(128, ext_routed_w1.shapes[2] - ffn_routed_w1_inline23497_offsets[2]))};
            Tensor ffn_routed_w1_inline23497 = ext_routed_w1.view(ffn_routed_w1_inline23497_shapes, ffn_routed_w1_inline23497_offsets);
            uint32_t ffn_routed_w1_scale_inline23498_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 10240)), 0};
            uint32_t ffn_routed_w1_scale_inline23498_shapes[2] = {(ffn_routed_w1_scale_inline23498_offsets[0] >= ext_routed_w1_scale.shapes[0] ? 0u : std::min<uint32_t>(10240, ext_routed_w1_scale.shapes[0] - ffn_routed_w1_scale_inline23498_offsets[0])), (ffn_routed_w1_scale_inline23498_offsets[1] >= ext_routed_w1_scale.shapes[1] ? 0u : std::min<uint32_t>(2304, ext_routed_w1_scale.shapes[1] - ffn_routed_w1_scale_inline23498_offsets[1]))};
            Tensor ffn_routed_w1_scale_inline23498 = ext_routed_w1_scale.view(ffn_routed_w1_scale_inline23498_shapes, ffn_routed_w1_scale_inline23498_offsets);
            uint32_t ffn_routed_w2_inline23499_offsets[3] = {static_cast<uint32_t>(ffn_routed_base_inline23490), 0, 0};
            uint32_t ffn_routed_w2_inline23499_shapes[3] = {(ffn_routed_w2_inline23499_offsets[0] >= ext_routed_w2.shapes[0] ? 0u : std::min<uint32_t>(64, ext_routed_w2.shapes[0] - ffn_routed_w2_inline23499_offsets[0])), (ffn_routed_w2_inline23499_offsets[1] >= ext_routed_w2.shapes[1] ? 0u : std::min<uint32_t>(46080, ext_routed_w2.shapes[1] - ffn_routed_w2_inline23499_offsets[1])), (ffn_routed_w2_inline23499_offsets[2] >= ext_routed_w2.shapes[2] ? 0u : std::min<uint32_t>(128, ext_routed_w2.shapes[2] - ffn_routed_w2_inline23499_offsets[2]))};
            Tensor ffn_routed_w2_inline23499 = ext_routed_w2.view(ffn_routed_w2_inline23499_shapes, ffn_routed_w2_inline23499_offsets);
            uint32_t ffn_routed_w2_scale_inline23500_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 4608)), 0};
            uint32_t ffn_routed_w2_scale_inline23500_shapes[2] = {(ffn_routed_w2_scale_inline23500_offsets[0] >= ext_routed_w2_scale.shapes[0] ? 0u : std::min<uint32_t>(4608, ext_routed_w2_scale.shapes[0] - ffn_routed_w2_scale_inline23500_offsets[0])), (ffn_routed_w2_scale_inline23500_offsets[1] >= ext_routed_w2_scale.shapes[1] ? 0u : std::min<uint32_t>(5120, ext_routed_w2_scale.shapes[1] - ffn_routed_w2_scale_inline23500_offsets[1]))};
            Tensor ffn_routed_w2_scale_inline23500 = ext_routed_w2_scale.view(ffn_routed_w2_scale_inline23500_shapes, ffn_routed_w2_scale_inline23500_offsets);
            uint32_t ffn_routed_w3_inline23501_offsets[3] = {static_cast<uint32_t>(ffn_routed_base_inline23490), 0, 0};
            uint32_t ffn_routed_w3_inline23501_shapes[3] = {(ffn_routed_w3_inline23501_offsets[0] >= ext_routed_w3.shapes[0] ? 0u : std::min<uint32_t>(64, ext_routed_w3.shapes[0] - ffn_routed_w3_inline23501_offsets[0])), (ffn_routed_w3_inline23501_offsets[1] >= ext_routed_w3.shapes[1] ? 0u : std::min<uint32_t>(46080, ext_routed_w3.shapes[1] - ffn_routed_w3_inline23501_offsets[1])), (ffn_routed_w3_inline23501_offsets[2] >= ext_routed_w3.shapes[2] ? 0u : std::min<uint32_t>(128, ext_routed_w3.shapes[2] - ffn_routed_w3_inline23501_offsets[2]))};
            Tensor ffn_routed_w3_inline23501 = ext_routed_w3.view(ffn_routed_w3_inline23501_shapes, ffn_routed_w3_inline23501_offsets);
            uint32_t ffn_routed_w3_scale_inline23502_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 10240)), 0};
            uint32_t ffn_routed_w3_scale_inline23502_shapes[2] = {(ffn_routed_w3_scale_inline23502_offsets[0] >= ext_routed_w3_scale.shapes[0] ? 0u : std::min<uint32_t>(10240, ext_routed_w3_scale.shapes[0] - ffn_routed_w3_scale_inline23502_offsets[0])), (ffn_routed_w3_scale_inline23502_offsets[1] >= ext_routed_w3_scale.shapes[1] ? 0u : std::min<uint32_t>(2304, ext_routed_w3_scale.shapes[1] - ffn_routed_w3_scale_inline23502_offsets[1]))};
            Tensor ffn_routed_w3_scale_inline23502 = ext_routed_w3_scale.view(ffn_routed_w3_scale_inline23502_shapes, ffn_routed_w3_scale_inline23502_offsets);
            uint32_t ffn_shared_w1_inline23503_offsets[3] = {static_cast<uint32_t>(layer_inline23452), 0, 0};
            uint32_t ffn_shared_w1_inline23503_shapes[3] = {(ffn_shared_w1_inline23503_offsets[0] >= ext_shared_w1.shapes[0] ? 0u : std::min<uint32_t>(1, ext_shared_w1.shapes[0] - ffn_shared_w1_inline23503_offsets[0])), (ffn_shared_w1_inline23503_offsets[1] >= ext_shared_w1.shapes[1] ? 0u : std::min<uint32_t>(5120, ext_shared_w1.shapes[1] - ffn_shared_w1_inline23503_offsets[1])), (ffn_shared_w1_inline23503_offsets[2] >= ext_shared_w1.shapes[2] ? 0u : std::min<uint32_t>(2304, ext_shared_w1.shapes[2] - ffn_shared_w1_inline23503_offsets[2]))};
            Tensor ffn_shared_w1_inline23503_view = ext_shared_w1.view(ffn_shared_w1_inline23503_shapes, ffn_shared_w1_inline23503_offsets);
            bool ffn_shared_w1_inline23503_empty = ffn_shared_w1_inline23503_view.numel() == 0;
            Tensor ffn_shared_w1_inline23503 = ffn_shared_w1_inline23503_view;
            ffn_shared_w1_inline23503.ndims = 2;
            ffn_shared_w1_inline23503.shapes[0] = ffn_shared_w1_inline23503_view.shapes[1];
            ffn_shared_w1_inline23503.strides[0] = ffn_shared_w1_inline23503_view.strides[1];
            ffn_shared_w1_inline23503.shapes[1] = ffn_shared_w1_inline23503_view.shapes[2];
            ffn_shared_w1_inline23503.strides[1] = ffn_shared_w1_inline23503_view.strides[2];
            if (ffn_shared_w1_inline23503_empty) ffn_shared_w1_inline23503.shapes[0] = 0;
            uint64_t ffn_shared_w1_inline23503_expected_stride = 1;
            bool ffn_shared_w1_inline23503_contiguous = true;
            for (int32_t i = static_cast<int32_t>(ffn_shared_w1_inline23503.ndims) - 1; i >= 0; --i) {
              ffn_shared_w1_inline23503_contiguous = ffn_shared_w1_inline23503_contiguous && (ffn_shared_w1_inline23503.strides[i] == ffn_shared_w1_inline23503_expected_stride);
              ffn_shared_w1_inline23503_expected_stride *= ffn_shared_w1_inline23503.shapes[i];
            }
            ffn_shared_w1_inline23503.is_contiguous = ffn_shared_w1_inline23503_empty || ffn_shared_w1_inline23503_contiguous;
            uint32_t ffn_shared_w1_scale_inline23504_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 160)), 0};
            uint32_t ffn_shared_w1_scale_inline23504_shapes[2] = {(ffn_shared_w1_scale_inline23504_offsets[0] >= ext_shared_w1_scale.shapes[0] ? 0u : std::min<uint32_t>(160, ext_shared_w1_scale.shapes[0] - ffn_shared_w1_scale_inline23504_offsets[0])), (ffn_shared_w1_scale_inline23504_offsets[1] >= ext_shared_w1_scale.shapes[1] ? 0u : std::min<uint32_t>(2304, ext_shared_w1_scale.shapes[1] - ffn_shared_w1_scale_inline23504_offsets[1]))};
            Tensor ffn_shared_w1_scale_inline23504 = ext_shared_w1_scale.view(ffn_shared_w1_scale_inline23504_shapes, ffn_shared_w1_scale_inline23504_offsets);
            uint32_t ffn_shared_w2_inline23505_offsets[3] = {static_cast<uint32_t>(layer_inline23452), 0, 0};
            uint32_t ffn_shared_w2_inline23505_shapes[3] = {(ffn_shared_w2_inline23505_offsets[0] >= ext_shared_w2.shapes[0] ? 0u : std::min<uint32_t>(1, ext_shared_w2.shapes[0] - ffn_shared_w2_inline23505_offsets[0])), (ffn_shared_w2_inline23505_offsets[1] >= ext_shared_w2.shapes[1] ? 0u : std::min<uint32_t>(2304, ext_shared_w2.shapes[1] - ffn_shared_w2_inline23505_offsets[1])), (ffn_shared_w2_inline23505_offsets[2] >= ext_shared_w2.shapes[2] ? 0u : std::min<uint32_t>(5120, ext_shared_w2.shapes[2] - ffn_shared_w2_inline23505_offsets[2]))};
            Tensor ffn_shared_w2_inline23505_view = ext_shared_w2.view(ffn_shared_w2_inline23505_shapes, ffn_shared_w2_inline23505_offsets);
            bool ffn_shared_w2_inline23505_empty = ffn_shared_w2_inline23505_view.numel() == 0;
            Tensor ffn_shared_w2_inline23505 = ffn_shared_w2_inline23505_view;
            ffn_shared_w2_inline23505.ndims = 2;
            ffn_shared_w2_inline23505.shapes[0] = ffn_shared_w2_inline23505_view.shapes[1];
            ffn_shared_w2_inline23505.strides[0] = ffn_shared_w2_inline23505_view.strides[1];
            ffn_shared_w2_inline23505.shapes[1] = ffn_shared_w2_inline23505_view.shapes[2];
            ffn_shared_w2_inline23505.strides[1] = ffn_shared_w2_inline23505_view.strides[2];
            if (ffn_shared_w2_inline23505_empty) ffn_shared_w2_inline23505.shapes[0] = 0;
            uint64_t ffn_shared_w2_inline23505_expected_stride = 1;
            bool ffn_shared_w2_inline23505_contiguous = true;
            for (int32_t i = static_cast<int32_t>(ffn_shared_w2_inline23505.ndims) - 1; i >= 0; --i) {
              ffn_shared_w2_inline23505_contiguous = ffn_shared_w2_inline23505_contiguous && (ffn_shared_w2_inline23505.strides[i] == ffn_shared_w2_inline23505_expected_stride);
              ffn_shared_w2_inline23505_expected_stride *= ffn_shared_w2_inline23505.shapes[i];
            }
            ffn_shared_w2_inline23505.is_contiguous = ffn_shared_w2_inline23505_empty || ffn_shared_w2_inline23505_contiguous;
            uint32_t ffn_shared_w2_scale_inline23506_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 72)), 0};
            uint32_t ffn_shared_w2_scale_inline23506_shapes[2] = {(ffn_shared_w2_scale_inline23506_offsets[0] >= ext_shared_w2_scale.shapes[0] ? 0u : std::min<uint32_t>(72, ext_shared_w2_scale.shapes[0] - ffn_shared_w2_scale_inline23506_offsets[0])), (ffn_shared_w2_scale_inline23506_offsets[1] >= ext_shared_w2_scale.shapes[1] ? 0u : std::min<uint32_t>(5120, ext_shared_w2_scale.shapes[1] - ffn_shared_w2_scale_inline23506_offsets[1]))};
            Tensor ffn_shared_w2_scale_inline23506 = ext_shared_w2_scale.view(ffn_shared_w2_scale_inline23506_shapes, ffn_shared_w2_scale_inline23506_offsets);
            uint32_t ffn_shared_w3_inline23507_offsets[3] = {static_cast<uint32_t>(layer_inline23452), 0, 0};
            uint32_t ffn_shared_w3_inline23507_shapes[3] = {(ffn_shared_w3_inline23507_offsets[0] >= ext_shared_w3.shapes[0] ? 0u : std::min<uint32_t>(1, ext_shared_w3.shapes[0] - ffn_shared_w3_inline23507_offsets[0])), (ffn_shared_w3_inline23507_offsets[1] >= ext_shared_w3.shapes[1] ? 0u : std::min<uint32_t>(5120, ext_shared_w3.shapes[1] - ffn_shared_w3_inline23507_offsets[1])), (ffn_shared_w3_inline23507_offsets[2] >= ext_shared_w3.shapes[2] ? 0u : std::min<uint32_t>(2304, ext_shared_w3.shapes[2] - ffn_shared_w3_inline23507_offsets[2]))};
            Tensor ffn_shared_w3_inline23507_view = ext_shared_w3.view(ffn_shared_w3_inline23507_shapes, ffn_shared_w3_inline23507_offsets);
            bool ffn_shared_w3_inline23507_empty = ffn_shared_w3_inline23507_view.numel() == 0;
            Tensor ffn_shared_w3_inline23507 = ffn_shared_w3_inline23507_view;
            ffn_shared_w3_inline23507.ndims = 2;
            ffn_shared_w3_inline23507.shapes[0] = ffn_shared_w3_inline23507_view.shapes[1];
            ffn_shared_w3_inline23507.strides[0] = ffn_shared_w3_inline23507_view.strides[1];
            ffn_shared_w3_inline23507.shapes[1] = ffn_shared_w3_inline23507_view.shapes[2];
            ffn_shared_w3_inline23507.strides[1] = ffn_shared_w3_inline23507_view.strides[2];
            if (ffn_shared_w3_inline23507_empty) ffn_shared_w3_inline23507.shapes[0] = 0;
            uint64_t ffn_shared_w3_inline23507_expected_stride = 1;
            bool ffn_shared_w3_inline23507_contiguous = true;
            for (int32_t i = static_cast<int32_t>(ffn_shared_w3_inline23507.ndims) - 1; i >= 0; --i) {
              ffn_shared_w3_inline23507_contiguous = ffn_shared_w3_inline23507_contiguous && (ffn_shared_w3_inline23507.strides[i] == ffn_shared_w3_inline23507_expected_stride);
              ffn_shared_w3_inline23507_expected_stride *= ffn_shared_w3_inline23507.shapes[i];
            }
            ffn_shared_w3_inline23507.is_contiguous = ffn_shared_w3_inline23507_empty || ffn_shared_w3_inline23507_contiguous;
            uint32_t ffn_shared_w3_scale_inline23508_offsets[2] = {static_cast<uint32_t>((layer_inline23452 * 160)), 0};
            uint32_t ffn_shared_w3_scale_inline23508_shapes[2] = {(ffn_shared_w3_scale_inline23508_offsets[0] >= ext_shared_w3_scale.shapes[0] ? 0u : std::min<uint32_t>(160, ext_shared_w3_scale.shapes[0] - ffn_shared_w3_scale_inline23508_offsets[0])), (ffn_shared_w3_scale_inline23508_offsets[1] >= ext_shared_w3_scale.shapes[1] ? 0u : std::min<uint32_t>(2304, ext_shared_w3_scale.shapes[1] - ffn_shared_w3_scale_inline23508_offsets[1]))};
            Tensor ffn_shared_w3_scale_inline23508 = ext_shared_w3_scale.view(ffn_shared_w3_scale_inline23508_shapes, ffn_shared_w3_scale_inline23508_offsets);
            int64_t t_linear_inline27234 = (((LOCAL_T_DYN + 15) / 16) * 16);
            uint32_t x_flat_inline27235_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
            Tensor x_flat_inline27235 = attention_hidden_inline23443.reshape(x_flat_inline27235_shapes, 2);
            uint32_t inv_rms_inline27236_ci_shapes[2] = {static_cast<uint32_t>(t_linear_inline27234), 1};
            TensorCreateInfo inv_rms_inline27236_ci(inv_rms_inline27236_ci_shapes, 2, DataType::FLOAT32);
            TaskOutputTensors alloc_94 = alloc_tensors(inv_rms_inline27236_ci);
            const Tensor& inv_rms_inline27236 = alloc_94.get_ref(0);
            int64_t spmd_core_num_24 = ((LOCAL_T_DYN + 7) / 8);
            if ((0 < spmd_core_num_24)) {

                // Spmd mhc_rms_spmd_6: mhc_rms
                CoreTaskArgs params_t232;
                params_t232.add_input(x_flat_inline27235);
                params_t232.add_inout(inv_rms_inline27236);
                params_t232.add_scalar(LOCAL_T_DYN);
                params_t232.launch_spec.set_block_num(spmd_core_num_24);
                rt_submit_aiv_task(28, params_t232);
                inv_rms_inline27236__ssa_v1 = inv_rms_inline27236;
            } else {
                inv_rms_inline27236__ssa_v1 = inv_rms_inline27236;
            }
            uint32_t mixes_partials_inline27245_ci_shapes[2] = {static_cast<uint32_t>((t_linear_inline27234 * 4)), 32};
            TensorCreateInfo mixes_partials_inline27245_ci(mixes_partials_inline27245_ci_shapes, 2, DataType::FLOAT32);
            TaskOutputTensors alloc_95 = alloc_tensors(mixes_partials_inline27245_ci);
            const Tensor& mixes_partials_inline27245 = alloc_95.get_ref(0);
            int64_t tasks_inline27246 = (((LOCAL_T_DYN + 79) / 80) * 4);
            int32_t t__tmp_v1449 = rt_available_cluster_count();
            int64_t workers_inline27247 = std::min<int64_t>(tasks_inline27246, static_cast<int64_t>(t__tmp_v1449));
            if ((0 < workers_inline27247)) {

                // Spmd mhc_linear_6: _hc_linear
                CoreTaskArgs params_t233;
                params_t233.add_input(x_flat_inline27235);
                params_t233.add_input(ffn_hc_fn_inline23491);
                params_t233.add_inout(mixes_partials_inline27245);
                params_t233.add_scalar(LOCAL_T_DYN);
                params_t233.add_scalar(t_linear_inline27234);
                params_t233.launch_spec.set_block_num(workers_inline27247);
                rt_submit_aic_task(29, params_t233);
            } else {
            }
            Tensor partials_inline27231 = mixes_partials_inline27245;
            int64_t linear_tokens_inline27257 = (int64_t)inv_rms_inline27236__ssa_v1.shapes[0];
            uint32_t base_row_inline27258_shapes[2] = {1, 24};
            Tensor base_row_inline27258 = ffn_hc_base_inline23493.reshape(base_row_inline27258_shapes, 2);
            uint32_t residual_flat_inline27259_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 16};
            Tensor residual_flat_inline27259 = residual_mix_inline27210.reshape(residual_flat_inline27259_shapes, 2);
            uint32_t indices_scale0_inline27260[1] = {0};
            float scale0_inline27260 = get_tensor_data<float>(ffn_hc_scale_inline23492, 1, indices_scale0_inline27260);
            uint32_t indices_scale1_inline27261[1] = {1};
            float scale1_inline27261 = get_tensor_data<float>(ffn_hc_scale_inline23492, 1, indices_scale1_inline27261);
            uint32_t indices_scale2_inline27262[1] = {2};
            float scale2_inline27262 = get_tensor_data<float>(ffn_hc_scale_inline23492, 1, indices_scale2_inline27262);
            int64_t blocks_inline27263 = ((LOCAL_T_DYN + 1) / 2);
            int64_t workers_inline27264 = std::min<int64_t>(blocks_inline27263, 12);
            TaskId ret__tmp_v0_guarded_2_1 = TaskId::invalid();
            if ((0 < workers_inline27264)) {

                // Spmd mhc_sinkhorn_spmd_6: mhc_sinkhorn_2
                CoreTaskArgs params_t234;
                params_t234.add_input(partials_inline27231);
                params_t234.add_input(inv_rms_inline27236__ssa_v1);
                params_t234.add_input(base_row_inline27258);
                params_t234.add_inout(encoder_pre_mix__rv_v1);
                params_t234.add_inout(post_mix_inline27209);
                params_t234.add_inout(residual_flat_inline27259);
                params_t234.add_scalar(LOCAL_T_DYN);
                params_t234.add_scalar(blocks_inline27263);
                params_t234.add_scalar(workers_inline27264);
                params_t234.add_scalar(linear_tokens_inline27257);
                params_t234.add_scalar(to_u64(scale0_inline27260));
                params_t234.add_scalar(to_u64(scale1_inline27261));
                params_t234.add_scalar(to_u64(scale2_inline27262));
                params_t234.launch_spec.set_block_num(workers_inline27264);
                TaskOutputTensors task_234_outs = rt_submit_aiv_task(65, params_t234);
                TaskId ret__tmp_v0_nonempty_2_8 = task_234_outs.task_id();
                ret__tmp_v0_guarded_0_53 = encoder_pre_mix__rv_v1;
                ret__tmp_v0_guarded_1_39 = post_mix_inline27209;
                ret__tmp_v0_guarded_2_1 = ret__tmp_v0_nonempty_2_8;
            } else {

                // Phase-fence barrier 50: dependency-only dummy task
                CoreTaskArgs params_phase_fence_barrier_50;
                TaskOutputTensors phase_fence_barrier_50_outs = rt_submit_dummy_task(params_phase_fence_barrier_50);
                TaskId spmd_empty_tid_41 = phase_fence_barrier_50_outs.task_id();
                ret__tmp_v0_guarded_0_53 = encoder_pre_mix__rv_v1;
                ret__tmp_v0_guarded_1_39 = post_mix_inline27209;
                ret__tmp_v0_guarded_2_1 = spmd_empty_tid_41;
            }
            TaskId ready_inline27265 = ret__tmp_v0_guarded_2_1;
            TaskId mixes_ready_inline27232 = ready_inline27265;
            uint32_t x_flat_inline27369_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
            Tensor x_flat_inline27369 = attention_hidden_inline23443.reshape(x_flat_inline27369_shapes, 2);
            int64_t spmd_core_num_25 = ((LOCAL_T_DYN + 1) / 2);
            if ((0 < spmd_core_num_25)) {

                // Spmd hc_collapse_spmd_0: hc_collapse
                CoreTaskArgs params_t235;
                params_t235.add_input(attention_pre_mix_inline23444__rv_v2);
                params_t235.add_input(x_flat_inline27369);
                params_t235.add_inout(x_mixed_inline23450__rv_v1);
                params_t235.add_scalar(LOCAL_T_DYN);
                params_t235.launch_spec.set_block_num(spmd_core_num_25);
                params_t235.set_allow_early_resolve(true);
                TaskOutputTensors task_235_outs = rt_submit_aiv_task(66, params_t235);
                ret__tmp_v0_guarded_0_54 = x_mixed_inline23450__rv_v1;
            } else {

                // Phase-fence barrier 51: dependency-only dummy task
                CoreTaskArgs params_phase_fence_barrier_51;
                TaskOutputTensors phase_fence_barrier_51_outs = rt_submit_dummy_task(params_phase_fence_barrier_51);
                TaskId spmd_empty_tid_42 = phase_fence_barrier_51_outs.task_id();
                ret__tmp_v0_guarded_0_54 = x_mixed_inline23450__rv_v1;
            }
            TaskId mixes_ready_inline27211 = mixes_ready_inline27232;
            SIMPLER_SCOPE() {
                uint32_t x_norm_mx_inline27212_ci_shapes[2] = {80, 5120};
                TensorCreateInfo x_norm_mx_inline27212_ci(x_norm_mx_inline27212_ci_shapes, 2, DataType::FP8E4M3FN);
                uint32_t x_norm_scale_inline27213_ci_shapes[2] = {1, 12800};
                TensorCreateInfo x_norm_scale_inline27213_ci(x_norm_scale_inline27213_ci_shapes, 2, DataType::FP8E8M0);
                uint32_t indices_inline27214_ci_shapes[2] = {80, 6};
                TensorCreateInfo indices_inline27214_ci(indices_inline27214_ci_shapes, 2, DataType::INT32);
                uint32_t weights_inline27215_ci_shapes[2] = {80, 6};
                TensorCreateInfo weights_inline27215_ci(weights_inline27215_ci_shapes, 2, DataType::FLOAT32);
                uint32_t xg_buf_inline27393_ci_shapes[2] = {80, 5120};
                TensorCreateInfo xg_buf_inline27393_ci(xg_buf_inline27393_ci_shapes, 2, DataType::FLOAT32);
                uint32_t biased_scores_buf_inline27444_ci_shapes[2] = {80, 384};
                TensorCreateInfo biased_scores_buf_inline27444_ci(biased_scores_buf_inline27444_ci_shapes, 2, DataType::FLOAT32);
                uint32_t route_scores_buf_inline27450_ci_shapes[2] = {80, 384};
                TensorCreateInfo route_scores_buf_inline27450_ci(route_scores_buf_inline27450_ci_shapes, 2, DataType::FLOAT32);
                uint32_t shared_output_inline27217_ci_shapes[2] = {80, 5120};
                TensorCreateInfo shared_output_inline27217_ci(shared_output_inline27217_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t gate_fp32_inline27519_ci_shapes[2] = {80, 2304};
                TensorCreateInfo gate_fp32_inline27519_ci(gate_fp32_inline27519_ci_shapes, 2, DataType::FLOAT32);
                uint32_t up_fp32_inline27520_ci_shapes[2] = {80, 2304};
                TensorCreateInfo up_fp32_inline27520_ci(up_fp32_inline27520_ci_shapes, 2, DataType::FLOAT32);
                uint32_t h_tile_mx_inline27547_ci_shapes[2] = {80, 2304};
                TensorCreateInfo h_tile_mx_inline27547_ci(h_tile_mx_inline27547_ci_shapes, 2, DataType::FP8E4M3FN);
                uint32_t h_scale_backing_inline27548_ci_shapes[2] = {1, 5760};
                TensorCreateInfo h_scale_backing_inline27548_ci(h_scale_backing_inline27548_ci_shapes, 2, DataType::FP8E8M0);
                uint32_t recv_x_local_inline27218_ci_shapes[3] = {64, 160, 5120};
                TensorCreateInfo recv_x_local_inline27218_ci(recv_x_local_inline27218_ci_shapes, 3, DataType::FP8E4M3FN, /*manual_dep=*/true);
                uint32_t recv_scale_local_backing_inline27219_ci_shapes[2] = {1, 1638400};
                TensorCreateInfo recv_scale_local_backing_inline27219_ci(recv_scale_local_backing_inline27219_ci_shapes, 2, DataType::FP8E8M0, /*manual_dep=*/true);
                uint32_t recv_weight_local_inline27220_ci_shapes[2] = {64, 160};
                TensorCreateInfo recv_weight_local_inline27220_ci(recv_weight_local_inline27220_ci_shapes, 2, DataType::FLOAT32, /*manual_dep=*/true);
                uint32_t recv_route_local_inline27221_ci_shapes[2] = {64, 160};
                TensorCreateInfo recv_route_local_inline27221_ci(recv_route_local_inline27221_ci_shapes, 2, DataType::INT32, /*manual_dep=*/true);
                TaskOutputTensors alloc_96 = alloc_tensors(x_norm_mx_inline27212_ci, x_norm_scale_inline27213_ci, indices_inline27214_ci, weights_inline27215_ci, xg_buf_inline27393_ci, biased_scores_buf_inline27444_ci, route_scores_buf_inline27450_ci, shared_output_inline27217_ci, gate_fp32_inline27519_ci, up_fp32_inline27520_ci, h_tile_mx_inline27547_ci, h_scale_backing_inline27548_ci, recv_x_local_inline27218_ci, recv_scale_local_backing_inline27219_ci, recv_weight_local_inline27220_ci, recv_route_local_inline27221_ci);
                const Tensor& x_norm_mx_inline27212 = alloc_96.get_ref(0);
                const Tensor& x_norm_scale_inline27213 = alloc_96.get_ref(1);
                const Tensor& indices_inline27214 = alloc_96.get_ref(2);
                const Tensor& weights_inline27215 = alloc_96.get_ref(3);
                const Tensor& xg_buf_inline27393 = alloc_96.get_ref(4);
                const Tensor& biased_scores_buf_inline27444 = alloc_96.get_ref(5);
                const Tensor& route_scores_buf_inline27450 = alloc_96.get_ref(6);
                const Tensor& shared_output_inline27217 = alloc_96.get_ref(7);
                const Tensor& gate_fp32_inline27519 = alloc_96.get_ref(8);
                const Tensor& up_fp32_inline27520 = alloc_96.get_ref(9);
                const Tensor& h_tile_mx_inline27547 = alloc_96.get_ref(10);
                const Tensor& h_scale_backing_inline27548 = alloc_96.get_ref(11);
                const Tensor& recv_x_local_inline27218 = alloc_96.get_ref(12);
                const Tensor& recv_scale_local_backing_inline27219 = alloc_96.get_ref(13);
                const Tensor& recv_weight_local_inline27220 = alloc_96.get_ref(14);
                const Tensor& recv_route_local_inline27221 = alloc_96.get_ref(15);
                uint32_t recv_count_local_inline27222_ci_shapes[2] = {64, 1};
                TensorCreateInfo recv_count_local_inline27222_ci(recv_count_local_inline27222_ci_shapes, 2, DataType::INT32);
                uint32_t recv_meta_local_inline27223_ci_shapes[2] = {2, 64};
                TensorCreateInfo recv_meta_local_inline27223_ci(recv_meta_local_inline27223_ci_shapes, 2, DataType::INT32);
                uint32_t x_norm_mx_raw_inline27608_ci_shapes[2] = {80, 5120};
                TensorCreateInfo x_norm_mx_raw_inline27608_ci(x_norm_mx_raw_inline27608_ci_shapes, 2, DataType::INT8);
                uint32_t scale_src_inline27609_ci_shapes[2] = {80, 160};
                TensorCreateInfo scale_src_inline27609_ci(scale_src_inline27609_ci_shapes, 2, DataType::UINT8, /*manual_dep=*/true);
                uint32_t recv_scale_nd_inline27669_ci_shapes[2] = {10240, 160};
                TensorCreateInfo recv_scale_nd_inline27669_ci(recv_scale_nd_inline27669_ci_shapes, 2, DataType::FP8E8M0, /*manual_dep=*/true);
                uint32_t sublayer_inline27226_ci_shapes[2] = {80, 5120};
                TensorCreateInfo sublayer_inline27226_ci(sublayer_inline27226_ci_shapes, 2, DataType::BFLOAT16);
                TaskOutputTensors alloc_97 = alloc_tensors(recv_count_local_inline27222_ci, recv_meta_local_inline27223_ci, x_norm_mx_raw_inline27608_ci, scale_src_inline27609_ci, recv_scale_nd_inline27669_ci, sublayer_inline27226_ci);
                const Tensor& recv_count_local_inline27222 = alloc_97.get_ref(0);
                const Tensor& recv_meta_local_inline27223 = alloc_97.get_ref(1);
                const Tensor& x_norm_mx_raw_inline27608 = alloc_97.get_ref(2);
                const Tensor& scale_src_inline27609 = alloc_97.get_ref(3);
                const Tensor& recv_scale_nd_inline27669 = alloc_97.get_ref(4);
                const Tensor& sublayer_inline27226 = alloc_97.get_ref(5);
                int64_t active_tokens_inline27390 = static_cast<int64_t>(local_moe_tokens_inline23451);
                int64_t active_tokens_v1_inline27391 = std::max<int64_t>(active_tokens_inline27390, 0);
                int64_t active_tokens_v2_inline27392 = std::min<int64_t>(active_tokens_v1_inline27391, 80);

                // Spmd rms_norm_spmd_0: rms_norm
                CoreTaskArgs params_t236;
                params_t236.add_input(ret__tmp_v0_guarded_0_54);
                params_t236.add_inout(xg_buf_inline27393);
                params_t236.add_input(layer_ffn_norm_inline23494);
                params_t236.add_scalar(active_tokens_v2_inline27392);
                params_t236.launch_spec.set_block_num(10);
                rt_submit_aiv_task(67, params_t236);
                int64_t active_tokens_inline27416 = static_cast<int64_t>(local_moe_tokens_inline23451);
                int64_t active_tokens_inline27416__phi_v2;
                if ((active_tokens_inline27416 < 0)) {
                    active_tokens_inline27416__phi_v2 = 0;
                } else {
                    active_tokens_inline27416__phi_v2 = active_tokens_inline27416;
                }
                int64_t active_tokens_inline27416__phi_v4;
                if ((80 < active_tokens_inline27416__phi_v2)) {
                    active_tokens_inline27416__phi_v4 = 80;
                } else {
                    active_tokens_inline27416__phi_v4 = active_tokens_inline27416__phi_v2;
                }
                int64_t task_tokens_inline27417 = std::max<int64_t>(active_tokens_inline27416__phi_v4, 1);
                int64_t task_gate_tiles_inline27418 = ((task_tokens_inline27417 + 15) / 16);

                // Spmd x_norm_mx_quant_spmd_0: x_norm_mx_quant
                CoreTaskArgs params_t237;
                params_t237.add_inout(x_norm_mx_inline27212);
                params_t237.add_inout(x_norm_scale_inline27213);
                params_t237.add_input(xg_buf_inline27393);
                params_t237.launch_spec.set_block_num(25);
                rt_submit_aiv_task(68, params_t237);

                // Task 238: gate_pre_route
                CoreTaskArgs params_t238;
                params_t238.add_inout(indices_inline27214);
                params_t238.add_inout(weights_inline27215);
                params_t238.add_inout(biased_scores_buf_inline27444);
                params_t238.add_scalar(active_tokens_inline27416__phi_v4);
                rt_submit_aiv_task(69, params_t238);
                uint32_t gate_bias_matrix_inline27451_shapes[2] = {1, 128};
                Tensor gate_bias_matrix_inline27451 = ffn_correction_bias_inline23496.reshape(gate_bias_matrix_inline27451_shapes, 2);
                int64_t gate_tasks_inline27452 = (task_gate_tiles_inline27418 * 8);
                int32_t t__tmp_v1470 = rt_available_cluster_count();
                int64_t gate_workers_inline27453 = std::min<int64_t>(gate_tasks_inline27452, std::min<int64_t>(static_cast<int64_t>(t__tmp_v1470), 16));
                if ((0 < gate_workers_inline27453)) {

                    // Group gate: MixedKernels (AIC + AIV lanes)
                    CoreTaskArgs params_t239;
                    params_t239.add_inout(biased_scores_buf_inline27444);
                    params_t239.add_inout(route_scores_buf_inline27450);
                    params_t239.add_input(xg_buf_inline27393);
                    params_t239.add_input(ffn_gate_weight_inline23495);
                    params_t239.add_input(gate_bias_matrix_inline27451);
                    params_t239.add_scalar(gate_tasks_inline27452);
                    params_t239.add_scalar(gate_workers_inline27453);
                    MixedKernels mixed_239 = {70, 71, INVALID_KERNEL_ID};
                    params_t239.launch_spec.set_block_num(gate_workers_inline27453);
                    params_t239.launch_spec.set_require_sync_start(true);
                    rt_submit_task(mixed_239, params_t239);
                    ret__tmp_v0_guarded_0_55 = biased_scores_buf_inline27444;
                    ret__tmp_v0_guarded_1_40 = route_scores_buf_inline27450;
                } else {
                    ret__tmp_v0_guarded_0_55 = biased_scores_buf_inline27444;
                    ret__tmp_v0_guarded_1_40 = route_scores_buf_inline27450;
                }
                int64_t task_route_tiles_inline27489 = ((task_tokens_inline27417 + 7) / 8);

                // Spmd route_sort_spmd_0: route_sort
                CoreTaskArgs params_t240;
                params_t240.add_input(ret__tmp_v0_guarded_0_55);
                params_t240.add_input(ret__tmp_v0_guarded_1_40);
                params_t240.add_inout(indices_inline27214);
                params_t240.add_inout(weights_inline27215);
                params_t240.add_scalar(active_tokens_inline27416__phi_v4);
                params_t240.launch_spec.set_block_num(task_route_tiles_inline27489);
                rt_submit_aiv_task(72, params_t240);
                Tensor dispatch_indices_inline27216 = indices_inline27214;
                Tensor x_local_scale_mx_inline27518 = x_norm_scale_inline27213;
                for (int64_t gm_inline27521 = 0; gm_inline27521 < 1; gm_inline27521 += 1) {

                    // Spmd sh_gate_mx_mm_spmd_0: sh_gate_mx_mm
                    CoreTaskArgs params_t241;
                    params_t241.add_input(x_norm_mx_inline27212);
                    params_t241.add_input(x_local_scale_mx_inline27518);
                    params_t241.add_input(ffn_shared_w1_inline23503);
                    params_t241.add_input(ffn_shared_w1_scale_inline23504);
                    params_t241.add_inout(gate_fp32_inline27519);
                    params_t241.launch_spec.set_block_num(9);
                    rt_submit_aic_task(73, params_t241);

                    // Spmd sh_up_mx_mm_spmd_0: sh_gate_mx_mm
                    CoreTaskArgs params_t242;
                    params_t242.add_input(x_norm_mx_inline27212);
                    params_t242.add_input(x_local_scale_mx_inline27518);
                    params_t242.add_input(ffn_shared_w3_inline23507);
                    params_t242.add_input(ffn_shared_w3_scale_inline23508);
                    params_t242.add_inout(up_fp32_inline27520);
                    params_t242.launch_spec.set_block_num(9);
                    rt_submit_aic_task(73, params_t242);
                }
                for (int64_t mt_inline27549 = 0; mt_inline27549 < 5; mt_inline27549 += 1) {
                    uint32_t h_tile_fp32_inline27550_ci_shapes[2] = {16, 2304};
                    TensorCreateInfo h_tile_fp32_inline27550_ci(h_tile_fp32_inline27550_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_98 = alloc_tensors(h_tile_fp32_inline27550_ci);
                    const Tensor& h_tile_fp32_inline27550 = alloc_98.get_ref(0);
                    int64_t ts0_inline27522 = (mt_inline27549 * 16);

                    // Spmd sh_gate_up_act_spmd_0: sh_gate_up_act
                    CoreTaskArgs params_t243;
                    params_t243.add_inout(h_tile_fp32_inline27550);
                    params_t243.add_input(gate_fp32_inline27519);
                    params_t243.add_input(up_fp32_inline27520);
                    params_t243.add_inout(h_tile_mx_inline27547);
                    params_t243.add_inout(h_scale_backing_inline27548);
                    params_t243.add_scalar(ts0_inline27522);
                    params_t243.launch_spec.set_block_num(3);
                    rt_submit_aiv_task(74, params_t243);
                }
                Tensor shared_output_inline27217__rv_v2 = shared_output_inline27217;
                for (int64_t wm_inline27585 = 0; wm_inline27585 < 1; wm_inline27585 += 1) {
                    Tensor h_tile_scale_mx_inline27586 = h_scale_backing_inline27548;
                    int32_t t__tmp_v1490 = rt_available_cluster_count();
                    int64_t workers_inline27587 = std::min<int64_t>(static_cast<int64_t>(t__tmp_v1490), 10);
                    Tensor shared_output_inline27217__rv_v4 = ext_engram_lookup;
                    if ((0 < workers_inline27587)) {

                        // Group sh_w2_mm: MixedKernels (AIC + AIV lanes)
                        CoreTaskArgs params_t244;
                        params_t244.add_inout(shared_output_inline27217__rv_v2);
                        params_t244.add_input(h_tile_mx_inline27547);
                        params_t244.add_input(h_tile_scale_mx_inline27586);
                        params_t244.add_input(ffn_shared_w2_inline23505);
                        params_t244.add_input(ffn_shared_w2_scale_inline23506);
                        MixedKernels mixed_244 = {75, 76, INVALID_KERNEL_ID};
                        params_t244.launch_spec.set_block_num(workers_inline27587);
                        params_t244.launch_spec.set_require_sync_start(true);
                        rt_submit_task(mixed_244, params_t244);
                        shared_output_inline27217__rv_v4 = shared_output_inline27217__rv_v2;
                    } else {
                        shared_output_inline27217__rv_v4 = shared_output_inline27217__rv_v2;
                    }
                    shared_output_inline27217__rv_v2 = shared_output_inline27217__rv_v4;
                }

                // Spmd dispatch_stage_spmd_0: dispatch_stage
                CoreTaskArgs params_t245;
                params_t245.add_input(x_norm_mx_inline27212);
                params_t245.add_inout(x_norm_mx_raw_inline27608);
                params_t245.add_input(x_norm_scale_inline27213);
                params_t245.add_inout(scale_src_inline27609);
                params_t245.add_scalar(local_moe_tokens_inline23451);
                params_t245.launch_spec.set_block_num(10);
                TaskOutputTensors task_245_outs = rt_submit_aiv_task(77, params_t245);
                TaskId _stage_tid_inline27610 = task_245_outs.task_id();

                // Task 246: dispatch_meta_publish
                CoreTaskArgs params_t246;
                params_t246.add_input(dispatch_indices_inline27216);
                params_t246.add_inout(ext_recv_meta);
                params_t246.add_inout(ext_arrived);
                params_t246.add_scalar(local_moe_tokens_inline23451);
                params_t246.add_scalar(rank);
                params_t246.add_scalar(layer_inline23452);
                params_t246.add_scalar(recv_meta_ctx);
                params_t246.add_scalar(arrived_ctx);
                params_t246.set_allow_early_resolve(true);
                TaskOutputTensors task_246_outs = rt_submit_aiv_task(78, params_t246);
                TaskId _meta_push_tid_inline27625 = task_246_outs.task_id();

                // Task 247: dispatch_meta_wait
                CoreTaskArgs params_t247;
                params_t247.add_input(ext_arrived);
                params_t247.add_scalar(rank);
                params_t247.add_scalar(layer_inline23452);
                params_t247.add_scalar(arrived_ctx);
                TaskOutputTensors task_247_outs = rt_submit_aiv_task(79, params_t247);
                TaskId _meta_wait_tid_inline27640 = task_247_outs.task_id();

                // Task 248: dispatch_meta_finalize
                CoreTaskArgs params_t248;
                params_t248.add_input(ext_recv_meta);
                params_t248.add_inout(recv_meta_local_inline27223);
                params_t248.add_inout(recv_count_local_inline27222);
                params_t248.add_scalar(recv_meta_ctx);
                TaskId params_t248_deps[2];
                uint32_t params_t248_deps_count = 0;
                params_t248_deps[params_t248_deps_count++] = _meta_push_tid_inline27625;
                params_t248_deps[params_t248_deps_count++] = _meta_wait_tid_inline27640;
                params_t248.set_dependencies(params_t248_deps, params_t248_deps_count);
                params_t248.set_allow_early_resolve(true);
                TaskOutputTensors task_248_outs = rt_submit_aiv_task(80, params_t248);
                TaskId _meta_tid_inline27642 = task_248_outs.task_id();

                // Spmd dispatch_push_spmd_0: dispatch_push
                CoreTaskArgs params_t249;
                params_t249.add_input(dispatch_indices_inline27216);
                params_t249.add_input(weights_inline27215);
                params_t249.add_inout(ext_recv_x);
                params_t249.add_input(x_norm_mx_raw_inline27608);
                params_t249.add_inout(ext_recv_scale);
                params_t249.add_input(scale_src_inline27609);
                params_t249.add_inout(ext_recv_weights);
                params_t249.add_inout(ext_recv_routes);
                params_t249.add_scalar(local_moe_tokens_inline23451);
                params_t249.add_scalar(rank);
                params_t249.add_scalar(recv_x_ctx);
                params_t249.add_scalar(recv_scale_ctx);
                params_t249.add_scalar(recv_weights_ctx);
                params_t249.add_scalar(recv_routes_ctx);
                params_t249.launch_spec.set_block_num(128);
                TaskId params_t249_deps[1];
                uint32_t params_t249_deps_count = 0;
                params_t249_deps[params_t249_deps_count++] = _stage_tid_inline27610;
                params_t249.set_dependencies(params_t249_deps, params_t249_deps_count);
                TaskOutputTensors task_249_outs = rt_submit_aiv_task(81, params_t249);
                TaskId _push_tid_inline27647 = task_249_outs.task_id();

                // Task 250: combine_notify
                CoreTaskArgs params_t250;
                params_t250.add_inout(ext_data_arrived);
                params_t250.add_scalar(rank);
                params_t250.add_scalar(layer_inline23452);
                params_t250.add_scalar(data_arrived_ctx);
                TaskId params_t250_deps[1];
                uint32_t params_t250_deps_count = 0;
                params_t250_deps[params_t250_deps_count++] = _push_tid_inline27647;
                params_t250.set_dependencies(params_t250_deps, params_t250_deps_count);
                TaskOutputTensors task_250_outs = rt_submit_aiv_task(82, params_t250);
                TaskId _publish_tid_inline27664 = task_250_outs.task_id();

                // Task 251: combine_wait
                CoreTaskArgs params_t251;
                params_t251.add_input(ext_data_arrived);
                params_t251.add_scalar(rank);
                params_t251.add_scalar(layer_inline23452);
                params_t251.add_scalar(data_arrived_ctx);
                TaskId params_t251_deps[1];
                uint32_t params_t251_deps_count = 0;
                params_t251_deps[params_t251_deps_count++] = _publish_tid_inline27664;
                params_t251.set_dependencies(params_t251_deps, params_t251_deps_count);
                TaskOutputTensors task_251_outs = rt_submit_aiv_task(83, params_t251);
                TaskId _wait_tid_inline27666 = task_251_outs.task_id();
                uint32_t recv_x_out_flat_inline27668_shapes[2] = {10240, 5120};
                Tensor recv_x_out_flat_inline27668 = recv_x_local_inline27218.reshape(recv_x_out_flat_inline27668_shapes, 2);
                int32_t t__tmp_v1494 = rt_available_aiv_count();
                int64_t gather_workers_inline27671 = std::min<int64_t>(static_cast<int64_t>(t__tmp_v1494), 640);
                TaskId ret__tmp_v0_guarded_1_41 = TaskId::invalid();
                if ((0 < gather_workers_inline27671)) {

                    // Spmd dispatch_gather_spmd_0: dispatch_gather
                    CoreTaskArgs params_t252;
                    params_t252.add_inout(recv_scale_local_backing_inline27219);
                    params_t252.add_inout(recv_scale_nd_inline27669);
                    params_t252.add_inout(recv_x_out_flat_inline27668);
                    params_t252.add_input(recv_count_local_inline27222);
                    params_t252.add_inout(recv_weight_local_inline27220);
                    params_t252.add_input(recv_meta_local_inline27223);
                    params_t252.add_input(ext_recv_x);
                    params_t252.add_input(ext_recv_scale);
                    params_t252.add_input(ext_recv_weights);
                    params_t252.add_input(ext_recv_routes);
                    params_t252.add_inout(recv_route_local_inline27221);
                    params_t252.add_scalar(gather_workers_inline27671);
                    params_t252.add_scalar(recv_x_ctx);
                    params_t252.add_scalar(recv_scale_ctx);
                    params_t252.add_scalar(recv_weights_ctx);
                    params_t252.add_scalar(recv_routes_ctx);
                    params_t252.launch_spec.set_block_num(gather_workers_inline27671);
                    TaskId params_t252_deps[3];
                    uint32_t params_t252_deps_count = 0;
                    params_t252_deps[params_t252_deps_count++] = _wait_tid_inline27666;
                    params_t252_deps[params_t252_deps_count++] = _push_tid_inline27647;
                    params_t252_deps[params_t252_deps_count++] = _meta_tid_inline27642;
                    params_t252.set_dependencies(params_t252_deps, params_t252_deps_count);
                    TaskOutputTensors task_252_outs = rt_submit_aiv_task(84, params_t252);
                    TaskId ret__tmp_v0_nonempty_1_44 = task_252_outs.task_id();
                    ret__tmp_v0_guarded_0_56 = recv_scale_local_backing_inline27219;
                    ret__tmp_v0_guarded_1_41 = ret__tmp_v0_nonempty_1_44;
                } else {

                    // Phase-fence barrier 52: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_52;
                    TaskId params_phase_fence_barrier_52_deps[3];
                    uint32_t params_phase_fence_barrier_52_deps_count = 0;
                    params_phase_fence_barrier_52_deps[params_phase_fence_barrier_52_deps_count++] = _wait_tid_inline27666;
                    params_phase_fence_barrier_52_deps[params_phase_fence_barrier_52_deps_count++] = _push_tid_inline27647;
                    params_phase_fence_barrier_52_deps[params_phase_fence_barrier_52_deps_count++] = _meta_tid_inline27642;
                    params_phase_fence_barrier_52.set_dependencies(params_phase_fence_barrier_52_deps, params_phase_fence_barrier_52_deps_count);
                    TaskId spmd_empty_tid_43 = TaskId::invalid();
                    if (params_phase_fence_barrier_52_deps_count > 0) {
                        TaskOutputTensors phase_fence_barrier_52_outs = rt_submit_dummy_task(params_phase_fence_barrier_52);
                        spmd_empty_tid_43 = phase_fence_barrier_52_outs.task_id();
                    }
                    ret__tmp_v0_guarded_0_56 = recv_scale_local_backing_inline27219;
                    ret__tmp_v0_guarded_1_41 = spmd_empty_tid_43;
                }
                TaskId _gather_tid_inline27672 = ret__tmp_v0_guarded_1_41;
                TaskId input_ready_inline27224 = _gather_tid_inline27672;
                TaskId hidden_completion_tids_inline27709[64]; for (int64_t __init_i = 0; __init_i < 64; ++__init_i) hidden_completion_tids_inline27709[__init_i] = TaskId::invalid();
                TaskId expert_completion_tids_inline27710[64]; for (int64_t __init_i = 0; __init_i < 64; ++__init_i) expert_completion_tids_inline27710[__init_i] = TaskId::invalid();
                uint32_t recv_x_flat_inline27711_shapes[2] = {10240, 5120};
                Tensor recv_x_flat_inline27711 = recv_x_local_inline27218.reshape(recv_x_flat_inline27711_shapes, 2);
                Tensor recv_mx_scale_view_inline27712 = ret__tmp_v0_guarded_0_56;
                SIMPLER_SCOPE() {
                    uint32_t h_mx_inline27713_ci_shapes[2] = {10240, 2304};
                    TensorCreateInfo h_mx_inline27713_ci(h_mx_inline27713_ci_shapes, 2, DataType::FP8E4M3FN);
                    uint32_t h_scale_backing_inline27714_ci_shapes[2] = {1, 737280};
                    TensorCreateInfo h_scale_backing_inline27714_ci(h_scale_backing_inline27714_ci_shapes, 2, DataType::FP8E8M0);
                    uint32_t w1_workspace_inline27715_ci_shapes[3] = {16, 5120, 2304};
                    TensorCreateInfo w1_workspace_inline27715_ci(w1_workspace_inline27715_ci_shapes, 3, DataType::FP8E4M3FN);
                    uint32_t w3_workspace_inline27716_ci_shapes[3] = {16, 5120, 2304};
                    TensorCreateInfo w3_workspace_inline27716_ci(w3_workspace_inline27716_ci_shapes, 3, DataType::FP8E4M3FN);
                    TaskOutputTensors alloc_99 = alloc_tensors(h_mx_inline27713_ci, h_scale_backing_inline27714_ci, w1_workspace_inline27715_ci, w3_workspace_inline27716_ci);
                    const Tensor& h_mx_inline27713 = alloc_99.get_ref(0);
                    const Tensor& h_scale_backing_inline27714 = alloc_99.get_ref(1);
                    const Tensor& w1_workspace_inline27715 = alloc_99.get_ref(2);
                    const Tensor& w3_workspace_inline27716 = alloc_99.get_ref(3);
                    TaskId gate_workspace_ready_inline27717[16]; for (int64_t __init_i = 0; __init_i < 16; ++__init_i) gate_workspace_ready_inline27717[__init_i] = TaskId::invalid();
                    gate_workspace_ready_inline27717[0] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[1] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[2] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[3] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[4] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[5] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[6] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[7] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[8] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[9] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[10] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[11] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[12] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[13] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[14] = input_ready_inline27224;
                    gate_workspace_ready_inline27717[15] = input_ready_inline27224;
                    for (int64_t local_i_inline27719 = 0; local_i_inline27719 < 64; local_i_inline27719 += 1) {
                        int64_t workspace_slot_inline27720 = (local_i_inline27719 % 16);
                        int64_t flat_base_inline27721 = (local_i_inline27719 * 160);
                        uint32_t routed_w1_packed_e_inline27722_offsets[3] = {static_cast<uint32_t>(local_i_inline27719), 0, 0};
                        uint32_t routed_w1_packed_e_inline27722_shapes[3] = {(routed_w1_packed_e_inline27722_offsets[0] >= ffn_routed_w1_inline23497.shapes[0] ? 0u : std::min<uint32_t>(1, ffn_routed_w1_inline23497.shapes[0] - routed_w1_packed_e_inline27722_offsets[0])), (routed_w1_packed_e_inline27722_offsets[1] >= ffn_routed_w1_inline23497.shapes[1] ? 0u : std::min<uint32_t>(46080, ffn_routed_w1_inline23497.shapes[1] - routed_w1_packed_e_inline27722_offsets[1])), (routed_w1_packed_e_inline27722_offsets[2] >= ffn_routed_w1_inline23497.shapes[2] ? 0u : std::min<uint32_t>(128, ffn_routed_w1_inline23497.shapes[2] - routed_w1_packed_e_inline27722_offsets[2]))};
                        Tensor routed_w1_packed_e_inline27722_view = ffn_routed_w1_inline23497.view(routed_w1_packed_e_inline27722_shapes, routed_w1_packed_e_inline27722_offsets);
                        bool routed_w1_packed_e_inline27722_empty = routed_w1_packed_e_inline27722_view.numel() == 0;
                        Tensor routed_w1_packed_e_inline27722 = routed_w1_packed_e_inline27722_view;
                        routed_w1_packed_e_inline27722.ndims = 2;
                        routed_w1_packed_e_inline27722.shapes[0] = routed_w1_packed_e_inline27722_view.shapes[1];
                        routed_w1_packed_e_inline27722.strides[0] = routed_w1_packed_e_inline27722_view.strides[1];
                        routed_w1_packed_e_inline27722.shapes[1] = routed_w1_packed_e_inline27722_view.shapes[2];
                        routed_w1_packed_e_inline27722.strides[1] = routed_w1_packed_e_inline27722_view.strides[2];
                        if (routed_w1_packed_e_inline27722_empty) routed_w1_packed_e_inline27722.shapes[0] = 0;
                        uint64_t routed_w1_packed_e_inline27722_expected_stride = 1;
                        bool routed_w1_packed_e_inline27722_contiguous = true;
                        for (int32_t i = static_cast<int32_t>(routed_w1_packed_e_inline27722.ndims) - 1; i >= 0; --i) {
                          routed_w1_packed_e_inline27722_contiguous = routed_w1_packed_e_inline27722_contiguous && (routed_w1_packed_e_inline27722.strides[i] == routed_w1_packed_e_inline27722_expected_stride);
                          routed_w1_packed_e_inline27722_expected_stride *= routed_w1_packed_e_inline27722.shapes[i];
                        }
                        routed_w1_packed_e_inline27722.is_contiguous = routed_w1_packed_e_inline27722_empty || routed_w1_packed_e_inline27722_contiguous;
                        uint32_t routed_w3_packed_e_inline27723_offsets[3] = {static_cast<uint32_t>(local_i_inline27719), 0, 0};
                        uint32_t routed_w3_packed_e_inline27723_shapes[3] = {(routed_w3_packed_e_inline27723_offsets[0] >= ffn_routed_w3_inline23501.shapes[0] ? 0u : std::min<uint32_t>(1, ffn_routed_w3_inline23501.shapes[0] - routed_w3_packed_e_inline27723_offsets[0])), (routed_w3_packed_e_inline27723_offsets[1] >= ffn_routed_w3_inline23501.shapes[1] ? 0u : std::min<uint32_t>(46080, ffn_routed_w3_inline23501.shapes[1] - routed_w3_packed_e_inline27723_offsets[1])), (routed_w3_packed_e_inline27723_offsets[2] >= ffn_routed_w3_inline23501.shapes[2] ? 0u : std::min<uint32_t>(128, ffn_routed_w3_inline23501.shapes[2] - routed_w3_packed_e_inline27723_offsets[2]))};
                        Tensor routed_w3_packed_e_inline27723_view = ffn_routed_w3_inline23501.view(routed_w3_packed_e_inline27723_shapes, routed_w3_packed_e_inline27723_offsets);
                        bool routed_w3_packed_e_inline27723_empty = routed_w3_packed_e_inline27723_view.numel() == 0;
                        Tensor routed_w3_packed_e_inline27723 = routed_w3_packed_e_inline27723_view;
                        routed_w3_packed_e_inline27723.ndims = 2;
                        routed_w3_packed_e_inline27723.shapes[0] = routed_w3_packed_e_inline27723_view.shapes[1];
                        routed_w3_packed_e_inline27723.strides[0] = routed_w3_packed_e_inline27723_view.strides[1];
                        routed_w3_packed_e_inline27723.shapes[1] = routed_w3_packed_e_inline27723_view.shapes[2];
                        routed_w3_packed_e_inline27723.strides[1] = routed_w3_packed_e_inline27723_view.strides[2];
                        if (routed_w3_packed_e_inline27723_empty) routed_w3_packed_e_inline27723.shapes[0] = 0;
                        uint64_t routed_w3_packed_e_inline27723_expected_stride = 1;
                        bool routed_w3_packed_e_inline27723_contiguous = true;
                        for (int32_t i = static_cast<int32_t>(routed_w3_packed_e_inline27723.ndims) - 1; i >= 0; --i) {
                          routed_w3_packed_e_inline27723_contiguous = routed_w3_packed_e_inline27723_contiguous && (routed_w3_packed_e_inline27723.strides[i] == routed_w3_packed_e_inline27723_expected_stride);
                          routed_w3_packed_e_inline27723_expected_stride *= routed_w3_packed_e_inline27723.shapes[i];
                        }
                        routed_w3_packed_e_inline27723.is_contiguous = routed_w3_packed_e_inline27723_empty || routed_w3_packed_e_inline27723_contiguous;
                        SIMPLER_SCOPE() {
                            uint32_t gate_tile_bf16_inline27770_ci_shapes[2] = {160, 2304};
                            TensorCreateInfo gate_tile_bf16_inline27770_ci(gate_tile_bf16_inline27770_ci_shapes, 2, DataType::BFLOAT16);
                            uint32_t up_tile_bf16_inline27771_ci_shapes[2] = {160, 2304};
                            TensorCreateInfo up_tile_bf16_inline27771_ci(up_tile_bf16_inline27771_ci_shapes, 2, DataType::BFLOAT16);
                            TaskOutputTensors alloc_100 = alloc_tensors(gate_tile_bf16_inline27770_ci, up_tile_bf16_inline27771_ci);
                            const Tensor& gate_tile_bf16_inline27770 = alloc_100.get_ref(0);
                            const Tensor& up_tile_bf16_inline27771 = alloc_100.get_ref(1);
                            uint32_t w1_mxfp8_inline27725_offsets[3] = {static_cast<uint32_t>(workspace_slot_inline27720), 0, 0};
                            uint32_t w1_mxfp8_inline27725_shapes[3] = {(w1_mxfp8_inline27725_offsets[0] >= w1_workspace_inline27715.shapes[0] ? 0u : std::min<uint32_t>(1, w1_workspace_inline27715.shapes[0] - w1_mxfp8_inline27725_offsets[0])), (w1_mxfp8_inline27725_offsets[1] >= w1_workspace_inline27715.shapes[1] ? 0u : std::min<uint32_t>(5120, w1_workspace_inline27715.shapes[1] - w1_mxfp8_inline27725_offsets[1])), (w1_mxfp8_inline27725_offsets[2] >= w1_workspace_inline27715.shapes[2] ? 0u : std::min<uint32_t>(2304, w1_workspace_inline27715.shapes[2] - w1_mxfp8_inline27725_offsets[2]))};
                            Tensor w1_mxfp8_inline27725_view = w1_workspace_inline27715.view(w1_mxfp8_inline27725_shapes, w1_mxfp8_inline27725_offsets);
                            bool w1_mxfp8_inline27725_empty = w1_mxfp8_inline27725_view.numel() == 0;
                            Tensor w1_mxfp8_inline27725 = w1_mxfp8_inline27725_view;
                            w1_mxfp8_inline27725.ndims = 2;
                            w1_mxfp8_inline27725.shapes[0] = w1_mxfp8_inline27725_view.shapes[1];
                            w1_mxfp8_inline27725.strides[0] = w1_mxfp8_inline27725_view.strides[1];
                            w1_mxfp8_inline27725.shapes[1] = w1_mxfp8_inline27725_view.shapes[2];
                            w1_mxfp8_inline27725.strides[1] = w1_mxfp8_inline27725_view.strides[2];
                            if (w1_mxfp8_inline27725_empty) w1_mxfp8_inline27725.shapes[0] = 0;
                            uint64_t w1_mxfp8_inline27725_expected_stride = 1;
                            bool w1_mxfp8_inline27725_contiguous = true;
                            for (int32_t i = static_cast<int32_t>(w1_mxfp8_inline27725.ndims) - 1; i >= 0; --i) {
                              w1_mxfp8_inline27725_contiguous = w1_mxfp8_inline27725_contiguous && (w1_mxfp8_inline27725.strides[i] == w1_mxfp8_inline27725_expected_stride);
                              w1_mxfp8_inline27725_expected_stride *= w1_mxfp8_inline27725.shapes[i];
                            }
                            w1_mxfp8_inline27725.is_contiguous = w1_mxfp8_inline27725_empty || w1_mxfp8_inline27725_contiguous;
                            TaskId _submit_deps_buf_inline27726[2]; for (int64_t __init_i = 0; __init_i < 2; ++__init_i) _submit_deps_buf_inline27726[__init_i] = TaskId::invalid();
                            _submit_deps_buf_inline27726[0] = input_ready_inline27224;
                            TaskId t__tmp_v1496 = gate_workspace_ready_inline27717[workspace_slot_inline27720];
                            _submit_deps_buf_inline27726[1] = t__tmp_v1496;

                            // Spmd exp_gate_mxfp4_aiv_spmd_0: exp_gate_mxfp4_aiv
                            CoreTaskArgs params_t253;
                            params_t253.add_input(recv_count_local_inline27222);
                            params_t253.add_input(ext_mxfp4_pair_lut);
                            params_t253.add_inout(w1_mxfp8_inline27725);
                            params_t253.add_input(routed_w1_packed_e_inline27722);
                            params_t253.add_scalar(local_i_inline27719);
                            params_t253.launch_spec.set_block_num(6);
                            TaskId params_t253_deps[2];
                            uint32_t params_t253_deps_count = 0;
                            if (_submit_deps_buf_inline27726[0].is_valid()) params_t253_deps[params_t253_deps_count++] = _submit_deps_buf_inline27726[0];
                            if (_submit_deps_buf_inline27726[1].is_valid()) params_t253_deps[params_t253_deps_count++] = _submit_deps_buf_inline27726[1];
                            params_t253.set_dependencies(params_t253_deps, params_t253_deps_count);
                            TaskOutputTensors task_253_outs = rt_submit_aiv_task(85, params_t253);
                            TaskId gate_mxfp4_aiv_tid_inline27729 = task_253_outs.task_id();
                            uint32_t w3_mxfp8_inline27748_offsets[3] = {static_cast<uint32_t>(workspace_slot_inline27720), 0, 0};
                            uint32_t w3_mxfp8_inline27748_shapes[3] = {(w3_mxfp8_inline27748_offsets[0] >= w3_workspace_inline27716.shapes[0] ? 0u : std::min<uint32_t>(1, w3_workspace_inline27716.shapes[0] - w3_mxfp8_inline27748_offsets[0])), (w3_mxfp8_inline27748_offsets[1] >= w3_workspace_inline27716.shapes[1] ? 0u : std::min<uint32_t>(5120, w3_workspace_inline27716.shapes[1] - w3_mxfp8_inline27748_offsets[1])), (w3_mxfp8_inline27748_offsets[2] >= w3_workspace_inline27716.shapes[2] ? 0u : std::min<uint32_t>(2304, w3_workspace_inline27716.shapes[2] - w3_mxfp8_inline27748_offsets[2]))};
                            Tensor w3_mxfp8_inline27748_view = w3_workspace_inline27716.view(w3_mxfp8_inline27748_shapes, w3_mxfp8_inline27748_offsets);
                            bool w3_mxfp8_inline27748_empty = w3_mxfp8_inline27748_view.numel() == 0;
                            Tensor w3_mxfp8_inline27748 = w3_mxfp8_inline27748_view;
                            w3_mxfp8_inline27748.ndims = 2;
                            w3_mxfp8_inline27748.shapes[0] = w3_mxfp8_inline27748_view.shapes[1];
                            w3_mxfp8_inline27748.strides[0] = w3_mxfp8_inline27748_view.strides[1];
                            w3_mxfp8_inline27748.shapes[1] = w3_mxfp8_inline27748_view.shapes[2];
                            w3_mxfp8_inline27748.strides[1] = w3_mxfp8_inline27748_view.strides[2];
                            if (w3_mxfp8_inline27748_empty) w3_mxfp8_inline27748.shapes[0] = 0;
                            uint64_t w3_mxfp8_inline27748_expected_stride = 1;
                            bool w3_mxfp8_inline27748_contiguous = true;
                            for (int32_t i = static_cast<int32_t>(w3_mxfp8_inline27748.ndims) - 1; i >= 0; --i) {
                              w3_mxfp8_inline27748_contiguous = w3_mxfp8_inline27748_contiguous && (w3_mxfp8_inline27748.strides[i] == w3_mxfp8_inline27748_expected_stride);
                              w3_mxfp8_inline27748_expected_stride *= w3_mxfp8_inline27748.shapes[i];
                            }
                            w3_mxfp8_inline27748.is_contiguous = w3_mxfp8_inline27748_empty || w3_mxfp8_inline27748_contiguous;
                            TaskId _submit_deps_buf_inline27749[2]; for (int64_t __init_i = 0; __init_i < 2; ++__init_i) _submit_deps_buf_inline27749[__init_i] = TaskId::invalid();
                            _submit_deps_buf_inline27749[0] = input_ready_inline27224;
                            TaskId t__tmp_v1497 = gate_workspace_ready_inline27717[workspace_slot_inline27720];
                            _submit_deps_buf_inline27749[1] = t__tmp_v1497;

                            // Spmd exp_up_mxfp4_aiv_spmd_0: exp_gate_mxfp4_aiv
                            CoreTaskArgs params_t254;
                            params_t254.add_input(recv_count_local_inline27222);
                            params_t254.add_input(ext_mxfp4_pair_lut);
                            params_t254.add_inout(w3_mxfp8_inline27748);
                            params_t254.add_input(routed_w3_packed_e_inline27723);
                            params_t254.add_scalar(local_i_inline27719);
                            params_t254.launch_spec.set_block_num(6);
                            TaskId params_t254_deps[2];
                            uint32_t params_t254_deps_count = 0;
                            if (_submit_deps_buf_inline27749[0].is_valid()) params_t254_deps[params_t254_deps_count++] = _submit_deps_buf_inline27749[0];
                            if (_submit_deps_buf_inline27749[1].is_valid()) params_t254_deps[params_t254_deps_count++] = _submit_deps_buf_inline27749[1];
                            params_t254.set_dependencies(params_t254_deps, params_t254_deps_count);
                            TaskOutputTensors task_254_outs = rt_submit_aiv_task(85, params_t254);
                            TaskId up_mxfp4_aiv_tid_inline27752 = task_254_outs.task_id();

                            // Spmd exp_gate_up_mxfp4_aic_spmd_0: exp_gate_up_mxfp4_aic
                            CoreTaskArgs params_t255;
                            params_t255.add_inout(gate_tile_bf16_inline27770);
                            params_t255.add_inout(up_tile_bf16_inline27771);
                            params_t255.add_input(recv_count_local_inline27222);
                            params_t255.add_input(recv_x_flat_inline27711);
                            params_t255.add_input(recv_mx_scale_view_inline27712);
                            params_t255.add_input(w1_mxfp8_inline27725);
                            params_t255.add_input(ffn_routed_w1_scale_inline23498);
                            params_t255.add_input(w3_mxfp8_inline27748);
                            params_t255.add_input(ffn_routed_w3_scale_inline23502);
                            params_t255.add_scalar(flat_base_inline27721);
                            params_t255.add_scalar(local_i_inline27719);
                            params_t255.launch_spec.set_block_num(9);
                            TaskId params_t255_deps[3];
                            uint32_t params_t255_deps_count = 0;
                            params_t255_deps[params_t255_deps_count++] = gate_mxfp4_aiv_tid_inline27729;
                            params_t255_deps[params_t255_deps_count++] = up_mxfp4_aiv_tid_inline27752;
                            if (ret__tmp_v0_guarded_1_41.is_valid()) params_t255_deps[params_t255_deps_count++] = ret__tmp_v0_guarded_1_41;
                            params_t255.set_dependencies(params_t255_deps, params_t255_deps_count);
                            TaskOutputTensors task_255_outs = rt_submit_aic_task(86, params_t255);
                            TaskId gate_up_mxfp4_aic_tid_inline27773 = task_255_outs.task_id();
                            uint32_t h_expert_mx_inline27791_offsets[2] = {static_cast<uint32_t>(flat_base_inline27721), 0};
                            uint32_t h_expert_mx_inline27791_shapes[2] = {(h_expert_mx_inline27791_offsets[0] >= h_mx_inline27713.shapes[0] ? 0u : std::min<uint32_t>(160, h_mx_inline27713.shapes[0] - h_expert_mx_inline27791_offsets[0])), (h_expert_mx_inline27791_offsets[1] >= h_mx_inline27713.shapes[1] ? 0u : std::min<uint32_t>(2304, h_mx_inline27713.shapes[1] - h_expert_mx_inline27791_offsets[1]))};
                            Tensor h_expert_mx_inline27791 = h_mx_inline27713.view(h_expert_mx_inline27791_shapes, h_expert_mx_inline27791_offsets);
                            uint32_t h_expert_scale_backing_inline27792_offsets[2] = {0, static_cast<uint32_t>((flat_base_inline27721 * 72))};
                            uint32_t h_expert_scale_backing_inline27792_shapes[2] = {(h_expert_scale_backing_inline27792_offsets[0] >= h_scale_backing_inline27714.shapes[0] ? 0u : std::min<uint32_t>(1, h_scale_backing_inline27714.shapes[0] - h_expert_scale_backing_inline27792_offsets[0])), (h_expert_scale_backing_inline27792_offsets[1] >= h_scale_backing_inline27714.shapes[1] ? 0u : std::min<uint32_t>(11520, h_scale_backing_inline27714.shapes[1] - h_expert_scale_backing_inline27792_offsets[1]))};
                            Tensor h_expert_scale_backing_inline27792 = h_scale_backing_inline27714.view(h_expert_scale_backing_inline27792_shapes, h_expert_scale_backing_inline27792_offsets);

                            // Spmd exp_gate_up_act_quant_spmd_0: exp_gate_up_act_quant
                            CoreTaskArgs params_t256;
                            params_t256.add_inout(h_expert_mx_inline27791);
                            params_t256.add_inout(h_expert_scale_backing_inline27792);
                            params_t256.add_input(recv_count_local_inline27222);
                            params_t256.add_input(recv_weight_local_inline27220);
                            params_t256.add_input(gate_tile_bf16_inline27770);
                            params_t256.add_input(up_tile_bf16_inline27771);
                            params_t256.add_scalar(local_i_inline27719);
                            params_t256.launch_spec.set_block_num(3);
                            TaskId params_t256_deps[1];
                            uint32_t params_t256_deps_count = 0;
                            params_t256_deps[params_t256_deps_count++] = gate_up_mxfp4_aic_tid_inline27773;
                            params_t256.set_dependencies(params_t256_deps, params_t256_deps_count);
                            TaskOutputTensors task_256_outs = rt_submit_aiv_task(87, params_t256);
                            TaskId gate_up_act_quant_tid_inline27794 = task_256_outs.task_id();
                            hidden_completion_tids_inline27709[local_i_inline27719] = gate_up_act_quant_tid_inline27794;
                            gate_workspace_ready_inline27717[workspace_slot_inline27720] = gate_up_act_quant_tid_inline27794;
                        }
                    }
                    SIMPLER_SCOPE() {
                        uint32_t w2_workspace_inline27830_ci_shapes[3] = {16, 2304, 5120};
                        TensorCreateInfo w2_workspace_inline27830_ci(w2_workspace_inline27830_ci_shapes, 3, DataType::FP8E4M3FN);
                        TaskOutputTensors alloc_101 = alloc_tensors(w2_workspace_inline27830_ci);
                        const Tensor& w2_workspace_inline27830 = alloc_101.get_ref(0);
                        TaskId down_workspace_ready_inline27831[16]; for (int64_t __init_i = 0; __init_i < 16; ++__init_i) down_workspace_ready_inline27831[__init_i] = TaskId::invalid();
                        down_workspace_ready_inline27831[0] = input_ready_inline27224;
                        down_workspace_ready_inline27831[1] = input_ready_inline27224;
                        down_workspace_ready_inline27831[2] = input_ready_inline27224;
                        down_workspace_ready_inline27831[3] = input_ready_inline27224;
                        down_workspace_ready_inline27831[4] = input_ready_inline27224;
                        down_workspace_ready_inline27831[5] = input_ready_inline27224;
                        down_workspace_ready_inline27831[6] = input_ready_inline27224;
                        down_workspace_ready_inline27831[7] = input_ready_inline27224;
                        down_workspace_ready_inline27831[8] = input_ready_inline27224;
                        down_workspace_ready_inline27831[9] = input_ready_inline27224;
                        down_workspace_ready_inline27831[10] = input_ready_inline27224;
                        down_workspace_ready_inline27831[11] = input_ready_inline27224;
                        down_workspace_ready_inline27831[12] = input_ready_inline27224;
                        down_workspace_ready_inline27831[13] = input_ready_inline27224;
                        down_workspace_ready_inline27831[14] = input_ready_inline27224;
                        down_workspace_ready_inline27831[15] = input_ready_inline27224;
                        for (int64_t local_e_inline27833 = 0; local_e_inline27833 < 64; local_e_inline27833 += 1) {
                            int64_t workspace_slot_inline27720__ssa_v1 = (local_e_inline27833 % 16);
                            int64_t e_flat_base_inline27834 = (local_e_inline27833 * 160);
                            uint32_t routed_w2_packed_e_inline27835_offsets[3] = {static_cast<uint32_t>(local_e_inline27833), 0, 0};
                            uint32_t routed_w2_packed_e_inline27835_shapes[3] = {(routed_w2_packed_e_inline27835_offsets[0] >= ffn_routed_w2_inline23499.shapes[0] ? 0u : std::min<uint32_t>(1, ffn_routed_w2_inline23499.shapes[0] - routed_w2_packed_e_inline27835_offsets[0])), (routed_w2_packed_e_inline27835_offsets[1] >= ffn_routed_w2_inline23499.shapes[1] ? 0u : std::min<uint32_t>(46080, ffn_routed_w2_inline23499.shapes[1] - routed_w2_packed_e_inline27835_offsets[1])), (routed_w2_packed_e_inline27835_offsets[2] >= ffn_routed_w2_inline23499.shapes[2] ? 0u : std::min<uint32_t>(128, ffn_routed_w2_inline23499.shapes[2] - routed_w2_packed_e_inline27835_offsets[2]))};
                            Tensor routed_w2_packed_e_inline27835_view = ffn_routed_w2_inline23499.view(routed_w2_packed_e_inline27835_shapes, routed_w2_packed_e_inline27835_offsets);
                            bool routed_w2_packed_e_inline27835_empty = routed_w2_packed_e_inline27835_view.numel() == 0;
                            Tensor routed_w2_packed_e_inline27835 = routed_w2_packed_e_inline27835_view;
                            routed_w2_packed_e_inline27835.ndims = 2;
                            routed_w2_packed_e_inline27835.shapes[0] = routed_w2_packed_e_inline27835_view.shapes[1];
                            routed_w2_packed_e_inline27835.strides[0] = routed_w2_packed_e_inline27835_view.strides[1];
                            routed_w2_packed_e_inline27835.shapes[1] = routed_w2_packed_e_inline27835_view.shapes[2];
                            routed_w2_packed_e_inline27835.strides[1] = routed_w2_packed_e_inline27835_view.strides[2];
                            if (routed_w2_packed_e_inline27835_empty) routed_w2_packed_e_inline27835.shapes[0] = 0;
                            uint64_t routed_w2_packed_e_inline27835_expected_stride = 1;
                            bool routed_w2_packed_e_inline27835_contiguous = true;
                            for (int32_t i = static_cast<int32_t>(routed_w2_packed_e_inline27835.ndims) - 1; i >= 0; --i) {
                              routed_w2_packed_e_inline27835_contiguous = routed_w2_packed_e_inline27835_contiguous && (routed_w2_packed_e_inline27835.strides[i] == routed_w2_packed_e_inline27835_expected_stride);
                              routed_w2_packed_e_inline27835_expected_stride *= routed_w2_packed_e_inline27835.shapes[i];
                            }
                            routed_w2_packed_e_inline27835.is_contiguous = routed_w2_packed_e_inline27835_empty || routed_w2_packed_e_inline27835_contiguous;
                            SIMPLER_SCOPE() {
                                uint32_t recv_y_tile_bf16_inline27859_ci_shapes[2] = {160, 5120};
                                TensorCreateInfo recv_y_tile_bf16_inline27859_ci(recv_y_tile_bf16_inline27859_ci_shapes, 2, DataType::BFLOAT16);
                                TaskOutputTensors alloc_102 = alloc_tensors(recv_y_tile_bf16_inline27859_ci);
                                const Tensor& recv_y_tile_bf16_inline27859 = alloc_102.get_ref(0);
                                uint32_t w2_mxfp8_inline27837_offsets[3] = {static_cast<uint32_t>(workspace_slot_inline27720__ssa_v1), 0, 0};
                                uint32_t w2_mxfp8_inline27837_shapes[3] = {(w2_mxfp8_inline27837_offsets[0] >= w2_workspace_inline27830.shapes[0] ? 0u : std::min<uint32_t>(1, w2_workspace_inline27830.shapes[0] - w2_mxfp8_inline27837_offsets[0])), (w2_mxfp8_inline27837_offsets[1] >= w2_workspace_inline27830.shapes[1] ? 0u : std::min<uint32_t>(2304, w2_workspace_inline27830.shapes[1] - w2_mxfp8_inline27837_offsets[1])), (w2_mxfp8_inline27837_offsets[2] >= w2_workspace_inline27830.shapes[2] ? 0u : std::min<uint32_t>(5120, w2_workspace_inline27830.shapes[2] - w2_mxfp8_inline27837_offsets[2]))};
                                Tensor w2_mxfp8_inline27837_view = w2_workspace_inline27830.view(w2_mxfp8_inline27837_shapes, w2_mxfp8_inline27837_offsets);
                                bool w2_mxfp8_inline27837_empty = w2_mxfp8_inline27837_view.numel() == 0;
                                Tensor w2_mxfp8_inline27837 = w2_mxfp8_inline27837_view;
                                w2_mxfp8_inline27837.ndims = 2;
                                w2_mxfp8_inline27837.shapes[0] = w2_mxfp8_inline27837_view.shapes[1];
                                w2_mxfp8_inline27837.strides[0] = w2_mxfp8_inline27837_view.strides[1];
                                w2_mxfp8_inline27837.shapes[1] = w2_mxfp8_inline27837_view.shapes[2];
                                w2_mxfp8_inline27837.strides[1] = w2_mxfp8_inline27837_view.strides[2];
                                if (w2_mxfp8_inline27837_empty) w2_mxfp8_inline27837.shapes[0] = 0;
                                uint64_t w2_mxfp8_inline27837_expected_stride = 1;
                                bool w2_mxfp8_inline27837_contiguous = true;
                                for (int32_t i = static_cast<int32_t>(w2_mxfp8_inline27837.ndims) - 1; i >= 0; --i) {
                                  w2_mxfp8_inline27837_contiguous = w2_mxfp8_inline27837_contiguous && (w2_mxfp8_inline27837.strides[i] == w2_mxfp8_inline27837_expected_stride);
                                  w2_mxfp8_inline27837_expected_stride *= w2_mxfp8_inline27837.shapes[i];
                                }
                                w2_mxfp8_inline27837.is_contiguous = w2_mxfp8_inline27837_empty || w2_mxfp8_inline27837_contiguous;
                                TaskId _submit_deps_buf_inline27838[2]; for (int64_t __init_i = 0; __init_i < 2; ++__init_i) _submit_deps_buf_inline27838[__init_i] = TaskId::invalid();
                                _submit_deps_buf_inline27838[0] = input_ready_inline27224;
                                TaskId t__tmp_v1498 = down_workspace_ready_inline27831[workspace_slot_inline27720__ssa_v1];
                                _submit_deps_buf_inline27838[1] = t__tmp_v1498;

                                // Spmd exp_w2_mxfp4_aiv_spmd_0: exp_w2_mxfp4_aiv
                                CoreTaskArgs params_t257;
                                params_t257.add_input(recv_count_local_inline27222);
                                params_t257.add_input(ext_mxfp4_pair_lut);
                                params_t257.add_inout(w2_mxfp8_inline27837);
                                params_t257.add_input(routed_w2_packed_e_inline27835);
                                params_t257.add_scalar(local_e_inline27833);
                                params_t257.launch_spec.set_block_num(10);
                                TaskId params_t257_deps[2];
                                uint32_t params_t257_deps_count = 0;
                                if (_submit_deps_buf_inline27838[0].is_valid()) params_t257_deps[params_t257_deps_count++] = _submit_deps_buf_inline27838[0];
                                if (_submit_deps_buf_inline27838[1].is_valid()) params_t257_deps[params_t257_deps_count++] = _submit_deps_buf_inline27838[1];
                                params_t257.set_dependencies(params_t257_deps, params_t257_deps_count);
                                TaskOutputTensors task_257_outs = rt_submit_aiv_task(88, params_t257);
                                TaskId w2_mxfp4_aiv_tid_inline27841 = task_257_outs.task_id();
                                uint32_t h_expert_mx_inline27791__ssa_v7_offsets[2] = {static_cast<uint32_t>(e_flat_base_inline27834), 0};
                                uint32_t h_expert_mx_inline27791__ssa_v7_shapes[2] = {(h_expert_mx_inline27791__ssa_v7_offsets[0] >= h_mx_inline27713.shapes[0] ? 0u : std::min<uint32_t>(160, h_mx_inline27713.shapes[0] - h_expert_mx_inline27791__ssa_v7_offsets[0])), (h_expert_mx_inline27791__ssa_v7_offsets[1] >= h_mx_inline27713.shapes[1] ? 0u : std::min<uint32_t>(2304, h_mx_inline27713.shapes[1] - h_expert_mx_inline27791__ssa_v7_offsets[1]))};
                                Tensor h_expert_mx_inline27791__ssa_v7 = h_mx_inline27713.view(h_expert_mx_inline27791__ssa_v7_shapes, h_expert_mx_inline27791__ssa_v7_offsets);
                                uint32_t h_expert_scale_backing_inline27792__ssa_v7_offsets[2] = {0, static_cast<uint32_t>((e_flat_base_inline27834 * 72))};
                                uint32_t h_expert_scale_backing_inline27792__ssa_v7_shapes[2] = {(h_expert_scale_backing_inline27792__ssa_v7_offsets[0] >= h_scale_backing_inline27714.shapes[0] ? 0u : std::min<uint32_t>(1, h_scale_backing_inline27714.shapes[0] - h_expert_scale_backing_inline27792__ssa_v7_offsets[0])), (h_expert_scale_backing_inline27792__ssa_v7_offsets[1] >= h_scale_backing_inline27714.shapes[1] ? 0u : std::min<uint32_t>(11520, h_scale_backing_inline27714.shapes[1] - h_expert_scale_backing_inline27792__ssa_v7_offsets[1]))};
                                Tensor h_expert_scale_backing_inline27792__ssa_v7 = h_scale_backing_inline27714.view(h_expert_scale_backing_inline27792__ssa_v7_shapes, h_expert_scale_backing_inline27792__ssa_v7_offsets);
                                Tensor h_expert_scale_mx_inline27858 = h_expert_scale_backing_inline27792__ssa_v7;
                                TaskId _submit_deps_buf_inline27861[2]; for (int64_t __init_i = 0; __init_i < 2; ++__init_i) _submit_deps_buf_inline27861[__init_i] = TaskId::invalid();
                                TaskId t__tmp_v1499 = hidden_completion_tids_inline27709[local_e_inline27833];
                                _submit_deps_buf_inline27861[0] = t__tmp_v1499;
                                _submit_deps_buf_inline27861[1] = w2_mxfp4_aiv_tid_inline27841;

                                // Spmd exp_w2_mxfp4_aic_spmd_0: exp_w2_mxfp4_aic
                                CoreTaskArgs params_t258;
                                params_t258.add_inout(recv_y_tile_bf16_inline27859);
                                params_t258.add_input(recv_count_local_inline27222);
                                params_t258.add_input(w2_mxfp8_inline27837);
                                params_t258.add_input(h_expert_mx_inline27791__ssa_v7);
                                params_t258.add_input(h_expert_scale_mx_inline27858);
                                params_t258.add_input(ffn_routed_w2_scale_inline23500);
                                params_t258.add_scalar(local_e_inline27833);
                                params_t258.launch_spec.set_block_num(5);
                                TaskId params_t258_deps[2];
                                uint32_t params_t258_deps_count = 0;
                                if (_submit_deps_buf_inline27861[0].is_valid()) params_t258_deps[params_t258_deps_count++] = _submit_deps_buf_inline27861[0];
                                if (_submit_deps_buf_inline27861[1].is_valid()) params_t258_deps[params_t258_deps_count++] = _submit_deps_buf_inline27861[1];
                                params_t258.set_dependencies(params_t258_deps, params_t258_deps_count);
                                TaskOutputTensors task_258_outs = rt_submit_aic_task(89, params_t258);
                                TaskId w2_mxfp4_aic_tid_inline27864 = task_258_outs.task_id();

                                // Task 259: expert_tile_scatter
                                CoreTaskArgs params_t259;
                                params_t259.add_input(recv_count_local_inline27222);
                                params_t259.add_input(recv_meta_local_inline27223);
                                params_t259.add_input(recv_route_local_inline27221);
                                params_t259.add_no_dep(ext_routed_output);
                                params_t259.add_input(recv_y_tile_bf16_inline27859);
                                params_t259.add_scalar(local_e_inline27833);
                                params_t259.add_scalar(routed_output_ctx);
                                TaskId params_t259_deps[1];
                                uint32_t params_t259_deps_count = 0;
                                params_t259_deps[params_t259_deps_count++] = w2_mxfp4_aic_tid_inline27864;
                                params_t259.set_dependencies(params_t259_deps, params_t259_deps_count);
                                TaskOutputTensors task_259_outs = rt_submit_aiv_task(90, params_t259);
                                TaskId scatter_tid_inline27875 = task_259_outs.task_id();
                                expert_completion_tids_inline27710[local_e_inline27833] = scatter_tid_inline27875;
                                down_workspace_ready_inline27831[workspace_slot_inline27720__ssa_v1] = scatter_tid_inline27875;
                            }
                        }
                    }
                }
                TaskId scatter_group_tids_inline27888[1]; for (int64_t __init_i = 0; __init_i < 1; ++__init_i) scatter_group_tids_inline27888[__init_i] = TaskId::invalid();
                TaskId group_tids_inline27890[64]; for (int64_t __init_i = 0; __init_i < 64; ++__init_i) group_tids_inline27890[__init_i] = TaskId::invalid();
                for (int64_t lane_inline27891 = 0; lane_inline27891 < 64; lane_inline27891 += 1) {
                    int64_t expert_inline27892 = lane_inline27891;
                    TaskId t__tmp_v1500 = expert_completion_tids_inline27710[expert_inline27892];
                    group_tids_inline27890[lane_inline27891] = t__tmp_v1500;
                }

                // Phase-fence barrier 53: dependency-only dummy task
                CoreTaskArgs params_phase_fence_barrier_53;
                TaskId params_phase_fence_barrier_53_deps[64];
                uint32_t params_phase_fence_barrier_53_deps_count = 0;
                if (group_tids_inline27890[0].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[0];
                if (group_tids_inline27890[1].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[1];
                if (group_tids_inline27890[2].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[2];
                if (group_tids_inline27890[3].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[3];
                if (group_tids_inline27890[4].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[4];
                if (group_tids_inline27890[5].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[5];
                if (group_tids_inline27890[6].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[6];
                if (group_tids_inline27890[7].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[7];
                if (group_tids_inline27890[8].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[8];
                if (group_tids_inline27890[9].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[9];
                if (group_tids_inline27890[10].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[10];
                if (group_tids_inline27890[11].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[11];
                if (group_tids_inline27890[12].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[12];
                if (group_tids_inline27890[13].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[13];
                if (group_tids_inline27890[14].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[14];
                if (group_tids_inline27890[15].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[15];
                if (group_tids_inline27890[16].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[16];
                if (group_tids_inline27890[17].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[17];
                if (group_tids_inline27890[18].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[18];
                if (group_tids_inline27890[19].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[19];
                if (group_tids_inline27890[20].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[20];
                if (group_tids_inline27890[21].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[21];
                if (group_tids_inline27890[22].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[22];
                if (group_tids_inline27890[23].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[23];
                if (group_tids_inline27890[24].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[24];
                if (group_tids_inline27890[25].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[25];
                if (group_tids_inline27890[26].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[26];
                if (group_tids_inline27890[27].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[27];
                if (group_tids_inline27890[28].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[28];
                if (group_tids_inline27890[29].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[29];
                if (group_tids_inline27890[30].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[30];
                if (group_tids_inline27890[31].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[31];
                if (group_tids_inline27890[32].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[32];
                if (group_tids_inline27890[33].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[33];
                if (group_tids_inline27890[34].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[34];
                if (group_tids_inline27890[35].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[35];
                if (group_tids_inline27890[36].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[36];
                if (group_tids_inline27890[37].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[37];
                if (group_tids_inline27890[38].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[38];
                if (group_tids_inline27890[39].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[39];
                if (group_tids_inline27890[40].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[40];
                if (group_tids_inline27890[41].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[41];
                if (group_tids_inline27890[42].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[42];
                if (group_tids_inline27890[43].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[43];
                if (group_tids_inline27890[44].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[44];
                if (group_tids_inline27890[45].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[45];
                if (group_tids_inline27890[46].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[46];
                if (group_tids_inline27890[47].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[47];
                if (group_tids_inline27890[48].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[48];
                if (group_tids_inline27890[49].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[49];
                if (group_tids_inline27890[50].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[50];
                if (group_tids_inline27890[51].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[51];
                if (group_tids_inline27890[52].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[52];
                if (group_tids_inline27890[53].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[53];
                if (group_tids_inline27890[54].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[54];
                if (group_tids_inline27890[55].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[55];
                if (group_tids_inline27890[56].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[56];
                if (group_tids_inline27890[57].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[57];
                if (group_tids_inline27890[58].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[58];
                if (group_tids_inline27890[59].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[59];
                if (group_tids_inline27890[60].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[60];
                if (group_tids_inline27890[61].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[61];
                if (group_tids_inline27890[62].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[62];
                if (group_tids_inline27890[63].is_valid()) params_phase_fence_barrier_53_deps[params_phase_fence_barrier_53_deps_count++] = group_tids_inline27890[63];
                params_phase_fence_barrier_53.set_dependencies(params_phase_fence_barrier_53_deps, params_phase_fence_barrier_53_deps_count);
                TaskId t__tmp_v1501 = TaskId::invalid();
                if (params_phase_fence_barrier_53_deps_count > 0) {
                    TaskOutputTensors phase_fence_barrier_53_outs = rt_submit_dummy_task(params_phase_fence_barrier_53);
                    t__tmp_v1501 = phase_fence_barrier_53_outs.task_id();
                }
                scatter_group_tids_inline27888[0] = t__tmp_v1501;

                // Phase-fence barrier 54: dependency-only dummy task
                CoreTaskArgs params_phase_fence_barrier_54;
                TaskId params_phase_fence_barrier_54_deps[1];
                uint32_t params_phase_fence_barrier_54_deps_count = 0;
                if (scatter_group_tids_inline27888[0].is_valid()) params_phase_fence_barrier_54_deps[params_phase_fence_barrier_54_deps_count++] = scatter_group_tids_inline27888[0];
                params_phase_fence_barrier_54.set_dependencies(params_phase_fence_barrier_54_deps, params_phase_fence_barrier_54_deps_count);
                TaskId scatter_done_inline27893 = TaskId::invalid();
                if (params_phase_fence_barrier_54_deps_count > 0) {
                    TaskOutputTensors phase_fence_barrier_54_outs = rt_submit_dummy_task(params_phase_fence_barrier_54);
                    scatter_done_inline27893 = phase_fence_barrier_54_outs.task_id();
                }
                TaskId scatter_done_inline27225 = scatter_done_inline27893;

                // Task 260: combine_notify
                CoreTaskArgs params_t260;
                params_t260.add_inout(ext_combine_arrived);
                params_t260.add_scalar(rank);
                params_t260.add_scalar(layer_inline23452);
                params_t260.add_scalar(combine_arrived_ctx);
                TaskId params_t260_deps[1];
                uint32_t params_t260_deps_count = 0;
                if (scatter_done_inline27893.is_valid()) params_t260_deps[params_t260_deps_count++] = scatter_done_inline27893;
                params_t260.set_dependencies(params_t260_deps, params_t260_deps_count);
                TaskOutputTensors task_260_outs = rt_submit_aiv_task(82, params_t260);
                TaskId notify_tid_inline27894 = task_260_outs.task_id();

                // Task 261: combine_wait
                CoreTaskArgs params_t261;
                params_t261.add_input(ext_combine_arrived);
                params_t261.add_scalar(rank);
                params_t261.add_scalar(layer_inline23452);
                params_t261.add_scalar(combine_arrived_ctx);
                TaskId params_t261_deps[1];
                uint32_t params_t261_deps_count = 0;
                params_t261_deps[params_t261_deps_count++] = notify_tid_inline27894;
                params_t261.set_dependencies(params_t261_deps, params_t261_deps_count);
                TaskOutputTensors task_261_outs = rt_submit_aiv_task(83, params_t261);
                TaskId _cwait_tid_inline27896 = task_261_outs.task_id();

                // Spmd combine_reduce_spmd_0: combine_reduce
                CoreTaskArgs params_t262;
                params_t262.add_inout(sublayer_inline27226);
                params_t262.add_input(shared_output_inline27217__rv_v2);
                params_t262.add_input(ext_routed_output);
                params_t262.add_scalar(local_moe_tokens_inline23451);
                params_t262.add_scalar(routed_output_ctx);
                params_t262.launch_spec.set_block_num(40);
                TaskId params_t262_deps[1];
                uint32_t params_t262_deps_count = 0;
                params_t262_deps[params_t262_deps_count++] = _cwait_tid_inline27896;
                params_t262.set_dependencies(params_t262_deps, params_t262_deps_count);
                TaskOutputTensors task_262_outs = rt_submit_aiv_task(91, params_t262);
                TaskId _reduce_tid_inline27898 = task_262_outs.task_id();
                TaskId output_ready_inline27227 = _reduce_tid_inline27898;
                TaskId hc_post_deps_inline27228[2]; for (int64_t __init_i = 0; __init_i < 2; ++__init_i) hc_post_deps_inline27228[__init_i] = TaskId::invalid();
                hc_post_deps_inline27228[0] = output_ready_inline27227;
                hc_post_deps_inline27228[1] = mixes_ready_inline27211;
                int64_t t_dim_inline27908 = 80;
                uint32_t residual_flat_inline27909_shapes[2] = {static_cast<uint32_t>(t_dim_inline27908), 20480};
                Tensor residual_flat_inline27909 = attention_hidden_inline23443.reshape(residual_flat_inline27909_shapes, 2);
                uint32_t residual_mix_flat_inline27910_shapes[2] = {static_cast<uint32_t>(t_dim_inline27908), 16};
                Tensor residual_mix_flat_inline27910 = residual_mix_inline27210.reshape(residual_mix_flat_inline27910_shapes, 2);
                uint32_t output_flat_inline27911_shapes[2] = {static_cast<uint32_t>(t_dim_inline27908), 20480};
                Tensor output_flat_inline27911 = ext_x_hc.reshape(output_flat_inline27911_shapes, 2);
                int64_t total_work_inline27912 = (t_dim_inline27908 * 2);
                int64_t full_work_inline27913 = ((total_work_inline27912 / 64) * 64);
                int64_t tail_work_inline27914 = ((total_work_inline27912 - full_work_inline27913) * 4);
                int64_t workers_inline27915 = std::min<int64_t>((total_work_inline27912 * 4), 64);
                if ((0 < workers_inline27915)) {

                    // Spmd hc_post_spmd_6: hc_post_2
                    CoreTaskArgs params_t263;
                    params_t263.add_inout(output_flat_inline27911);
                    params_t263.add_input(sublayer_inline27226);
                    params_t263.add_input(residual_flat_inline27909);
                    params_t263.add_input(ret__tmp_v0_guarded_1_39);
                    params_t263.add_input(residual_mix_flat_inline27910);
                    params_t263.add_scalar(full_work_inline27913);
                    params_t263.add_scalar(tail_work_inline27914);
                    params_t263.launch_spec.set_block_num(workers_inline27915);
                    TaskId params_t263_deps[2];
                    uint32_t params_t263_deps_count = 0;
                    if (hc_post_deps_inline27228[0].is_valid()) params_t263_deps[params_t263_deps_count++] = hc_post_deps_inline27228[0];
                    if (hc_post_deps_inline27228[1].is_valid()) params_t263_deps[params_t263_deps_count++] = hc_post_deps_inline27228[1];
                    params_t263.set_dependencies(params_t263_deps, params_t263_deps_count);
                    TaskOutputTensors task_263_outs = rt_submit_aiv_task(92, params_t263);
                } else {

                    // Phase-fence barrier 55: dependency-only dummy task
                    CoreTaskArgs params_phase_fence_barrier_55;
                    TaskId params_phase_fence_barrier_55_deps[2];
                    uint32_t params_phase_fence_barrier_55_deps_count = 0;
                    if (hc_post_deps_inline27228[0].is_valid()) params_phase_fence_barrier_55_deps[params_phase_fence_barrier_55_deps_count++] = hc_post_deps_inline27228[0];
                    if (hc_post_deps_inline27228[1].is_valid()) params_phase_fence_barrier_55_deps[params_phase_fence_barrier_55_deps_count++] = hc_post_deps_inline27228[1];
                    params_phase_fence_barrier_55.set_dependencies(params_phase_fence_barrier_55_deps, params_phase_fence_barrier_55_deps_count);
                    TaskId spmd_empty_tid_44 = TaskId::invalid();
                    if (params_phase_fence_barrier_55_deps_count > 0) {
                        TaskOutputTensors phase_fence_barrier_55_outs = rt_submit_dummy_task(params_phase_fence_barrier_55);
                        spmd_empty_tid_44 = phase_fence_barrier_55_outs.task_id();
                    }
                }
            }
        }
        attention_pre_mix_inline23444__rv_v3 = attention_pre_mix_inline23444__rv_v2;
        topk_indices_inline23445__rv_v5 = topk_indices_inline23445__rv_v4;
        pool_block_ids_inline23448__rv_v3 = pool_block_ids_inline23448__rv_v2;
        encoder_pre_mix__rv_v1 = ret__tmp_v0_guarded_0_53;
        x_mixed_inline23450__rv_v1 = ret__tmp_v0_guarded_0_54;
    }

    // Task 264: moe_signal_clear
    CoreTaskArgs params_t264;
    params_t264.add_inout(ext_x_hc);
    params_t264.add_inout(ext_arrived);
    params_t264.add_inout(ext_data_arrived);
    params_t264.add_inout(ext_combine_arrived);
    params_t264.add_scalar(arrived_ctx);
    params_t264.add_scalar(data_arrived_ctx);
    params_t264.add_scalar(combine_arrived_ctx);
    rt_submit_aiv_task(120, params_t264);
    uint32_t x_flat_inline27970_shapes[2] = {static_cast<uint32_t>(LOCAL_T_DYN), 20480};
    Tensor x_flat_inline27970 = ext_x_hc.reshape(x_flat_inline27970_shapes, 2);
    int64_t spmd_core_num_26 = ((LOCAL_T_DYN + 1) / 2);
    Tensor ret__tmp_v0_guarded_0_57 = ext_engram_lookup;
    if ((0 < spmd_core_num_26)) {

        // Spmd hc_collapse_spmd_1: hc_collapse
        CoreTaskArgs params_t265;
        params_t265.add_input(encoder_pre_mix__rv_v1);
        params_t265.add_input(x_flat_inline27970);
        params_t265.add_output(collapsed_inline27966);
        params_t265.add_scalar(LOCAL_T_DYN);
        params_t265.launch_spec.set_block_num(spmd_core_num_26);
        params_t265.set_allow_early_resolve(true);
        TaskOutputTensors task_265_outs = rt_submit_aiv_task(121, params_t265);
        ret__tmp_v0_guarded_0_57 = collapsed_inline27966;
    } else {

        // Phase-fence barrier 56: dependency-only dummy task
        CoreTaskArgs params_phase_fence_barrier_56;
        TaskOutputTensors phase_fence_barrier_56_outs = rt_submit_dummy_task(params_phase_fence_barrier_56);
        TaskId spmd_empty_tid_45 = phase_fence_barrier_56_outs.task_id();
        ret__tmp_v0_guarded_0_57 = collapsed_inline27966;
    }
    Tensor collapsed_inline27966__ssa_v1 = ret__tmp_v0_guarded_0_57;
    int64_t spmd_core_num_27 = ((LOCAL_T_DYN + 7) / 8);
    Tensor normalized_inline27967__rv_v2 = ext_engram_lookup;
    if ((0 < spmd_core_num_27)) {

        // Spmd final_rms_norm_spmd: final_rms_norm
        CoreTaskArgs params_t266;
        params_t266.add_input(collapsed_inline27966__ssa_v1);
        params_t266.add_output(normalized_inline27967);
        params_t266.add_input(ext_final_norm_weight);
        params_t266.add_scalar(LOCAL_T_DYN);
        params_t266.launch_spec.set_block_num(spmd_core_num_27);
        rt_submit_aiv_task(122, params_t266);
        normalized_inline27967__rv_v2 = normalized_inline27967;
    } else {
        normalized_inline27967__rv_v2 = normalized_inline27967;
    }

    // Spmd lm_head_dispatch_push_spmd: lm_head_dispatch_push
    CoreTaskArgs params_t267;
    params_t267.add_input(ext_logit_row_indices);
    params_t267.add_inout(selected_hidden_inline28012);
    params_t267.add_input(normalized_inline27967__rv_v2);
    params_t267.add_output(ext_hidden_window);
    params_t267.add_scalar(LOCAL_T_DYN);
    params_t267.add_scalar(rank);
    params_t267.add_scalar(hidden_window_ctx);
    params_t267.launch_spec.set_block_num(16);
    TaskOutputTensors task_267_outs = rt_submit_aiv_task(123, params_t267);
    TaskId _dispatch_push_tid_inline28014 = task_267_outs.task_id();

    // Task 268: lm_head_combine_wait
    CoreTaskArgs params_t268;
    TaskId params_t268_deps[1];
    uint32_t params_t268_deps_count = 0;
    params_t268_deps[params_t268_deps_count++] = _dispatch_push_tid_inline28014;
    params_t268.set_dependencies(params_t268_deps, params_t268_deps_count);
    TaskOutputTensors task_268_outs = rt_submit_aiv_task(124, params_t268);
    TaskId _dwait_tid_inline28022 = task_268_outs.task_id();

    // Spmd lm_head_dispatch_gather_spmd: lm_head_dispatch_gather
    CoreTaskArgs params_t269;
    params_t269.add_input(ext_hidden_window);
    params_t269.add_output(owner_hiddens_inline28013);
    params_t269.add_scalar(hidden_window_ctx);
    params_t269.launch_spec.set_block_num(10);
    TaskId params_t269_deps[1];
    uint32_t params_t269_deps_count = 0;
    params_t269_deps[params_t269_deps_count++] = _dwait_tid_inline28022;
    params_t269.set_dependencies(params_t269_deps, params_t269_deps_count);
    TaskOutputTensors task_269_outs = rt_submit_aiv_task(125, params_t269);

    // Spmd lm_head_matmul_spmd: lm_head_matmul
    CoreTaskArgs params_t270;
    params_t270.add_output(logits_shards_inline28027);
    params_t270.add_input(owner_hiddens_inline28013);
    params_t270.add_input(ext_lm_head_weight);
    params_t270.launch_spec.set_block_num(24);
    rt_submit_aic_task(126, params_t270);

    // Spmd lm_head_combine_push_spmd: lm_head_combine_push
    CoreTaskArgs params_t271;
    params_t271.add_output(ext_logits_window);
    params_t271.add_input(logits_shards_inline28027);
    params_t271.add_scalar(rank);
    params_t271.add_scalar(logits_window_ctx);
    params_t271.launch_spec.set_block_num(24);
    TaskOutputTensors task_271_outs = rt_submit_aiv_task(127, params_t271);
    TaskId _push_tid_inline28046 = task_271_outs.task_id();

    // Task 272: lm_head_combine_wait
    CoreTaskArgs params_t272;
    TaskId params_t272_deps[1];
    uint32_t params_t272_deps_count = 0;
    params_t272_deps[params_t272_deps_count++] = _push_tid_inline28046;
    params_t272.set_dependencies(params_t272_deps, params_t272_deps_count);
    TaskOutputTensors task_272_outs = rt_submit_aiv_task(124, params_t272);
    TaskId _cwait_tid_inline28055 = task_272_outs.task_id();

    // Spmd lm_head_combine_gather_spmd: lm_head_combine_gather
    CoreTaskArgs params_t273;
    params_t273.add_output(ext_logits);
    params_t273.add_input(ext_logits_window);
    params_t273.add_scalar(logits_window_ctx);
    params_t273.launch_spec.set_block_num(24);
    TaskId params_t273_deps[1];
    uint32_t params_t273_deps_count = 0;
    params_t273_deps[params_t273_deps_count++] = _cwait_tid_inline28055;
    params_t273.set_dependencies(params_t273_deps, params_t273_deps_count);
    TaskOutputTensors task_273_outs = rt_submit_aiv_task(128, params_t273);
    TaskId _gather_tid_inline28057 = task_273_outs.task_id();

    // Task 274: lm_head_combine_wait
    CoreTaskArgs params_t274;
    TaskId params_t274_deps[1];
    uint32_t params_t274_deps_count = 0;
    params_t274_deps[params_t274_deps_count++] = _gather_tid_inline28057;
    params_t274.set_dependencies(params_t274_deps, params_t274_deps_count);
    TaskOutputTensors task_274_outs = rt_submit_aiv_task(124, params_t274);
    TaskId _readback_notify_tid_inline28065 = task_274_outs.task_id();

    // Task 275: lm_head_combine_wait
    CoreTaskArgs params_t275;
    TaskOutputTensors task_275_outs = rt_submit_aiv_task(124, params_t275);
    TaskId _readback_wait_tid_inline28067 = task_275_outs.task_id();

    // Task 276: lm_head_signal_retire
    CoreTaskArgs params_t276;
    params_t276.add_inout(ext_logits);
    TaskId params_t276_deps[2];
    uint32_t params_t276_deps_count = 0;
    params_t276_deps[params_t276_deps_count++] = _readback_notify_tid_inline28065;
    params_t276_deps[params_t276_deps_count++] = _readback_wait_tid_inline28067;
    params_t276.set_dependencies(params_t276_deps, params_t276_deps_count);
    TaskOutputTensors task_276_outs = rt_submit_aiv_task(129, params_t276);

    // Spmd lm_head_greedy_sample_spmd: lm_head_greedy_sample
    CoreTaskArgs params_t277;
    params_t277.add_input(ext_logits);
    params_t277.add_output(ext_sampled_ids);
    params_t277.launch_spec.set_block_num(16);
    rt_submit_aiv_task(130, params_t277);
    Tensor final_logits = ext_logits;
}

}  // extern "C"
