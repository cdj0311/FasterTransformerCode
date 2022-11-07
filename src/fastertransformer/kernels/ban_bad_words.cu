/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/kernels/ban_bad_words.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T>
__global__ void ban_bad_words(T*         logits,
                              const int* output_ids_buf,
                              const int* parent_ids_buf,
                              int        batch_size,
                              int        beam_width,
                              const int* bad_words,
                              size_t     bad_words_len,
                              bool       share_words,
                              int        id_offset,
                              int        vocab_size_padded,
                              size_t     step)
{
    const int id        = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y / beam_width;
    const int beam_idx  = blockIdx.y % beam_width;

    const int* base_bad_words         = share_words ? bad_words : bad_words + batch_idx * 2 * bad_words_len;
    const int* base_bad_words_offsets = base_bad_words + bad_words_len;

    if (id >= bad_words_len || base_bad_words_offsets[id] < 0) {
        return;
    }

    const int item_end   = base_bad_words_offsets[id];
    const int item_start = (id > 0) ? base_bad_words_offsets[id - 1] : 0;
    const int item_size  = item_end - item_start;

    /* The single-token case unconditionally bans the token */
    bool should_ban = item_size == 1;

    /* Multi-token case and enough previously generated tokens to look for a match */
    if (item_size > 1 && step >= item_size - 1) {
        should_ban             = true;
        int        parent_id   = beam_idx;
        const bool gather_beam = beam_width > 1;

        for (int token_idx = item_size - 2; token_idx >= 0; token_idx--) {
            const int previous_token = output_ids_buf[(step - (item_size - 1) + token_idx) * batch_size * beam_width
                                                      + id_offset + batch_idx * beam_width + parent_id];

            if (previous_token != base_bad_words[item_start + token_idx]) {
                should_ban = false;
                break;
            }
            if (gather_beam) {
                parent_id = parent_ids_buf[(step - (item_size - 1) + token_idx) * beam_width * batch_size + id_offset
                                           + batch_idx * beam_width + parent_id];

                if (parent_id < 0 || parent_id >= beam_width) {
                    should_ban = false;
                    break;
                }
            }
        }
    }

    if (should_ban) {
        int banned_token = base_bad_words[item_end - 1];
        if (0 < banned_token && banned_token < vocab_size_padded) {
            logits[batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token] =
                static_cast<T>(-INFINITY);
        }
    }
}

template<typename T>
void invokeBanBadWords(T*           logits,
                       const int*   output_ids_buf,
                       const int*   parent_ids_buf,
                       int          batch_size,
                       int          local_batch_size,
                       int          beam_width,
                       const int*   bad_words,
                       bool         share_words,
                       size_t       bad_words_len,
                       int          id_offset,
                       int          vocab_size_padded,
                       size_t       step,
                       cudaStream_t stream)
{
    dim3 block, grid;
    block.x = min(((bad_words_len + 32 - 1) / 32) * 32, 256UL);
    grid.x  = (bad_words_len + block.x - 1) / block.x;
    grid.y  = local_batch_size * beam_width;

    ban_bad_words<<<grid, block, 0, stream>>>(logits,
                                              output_ids_buf,
                                              parent_ids_buf,
                                              batch_size,
                                              beam_width,
                                              bad_words,
                                              bad_words_len,
                                              share_words,
                                              id_offset,
                                              vocab_size_padded,
                                              step);
    sync_check_cuda_error();
}

template void invokeBanBadWords(half*        logits,
                                const int*   output_ids_buf,
                                const int*   parent_ids_buf,
                                int          batch_size,
                                int          local_batch_size,
                                int          beam_width,
                                const int*   bad_words,
                                bool         share_words,
                                size_t       bad_words_len,
                                int          id_offset,
                                int          vocab_size_padded,
                                size_t       step,
                                cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBanBadWords(__nv_bfloat16* logits,
                                const int*     output_ids_buf,
                                const int*     parent_ids_buf,
                                int            batch_size,
                                int            local_batch_size,
                                int            beam_width,
                                const int*     bad_words,
                                bool           share_words,
                                size_t         bad_words_len,
                                int            id_offset,
                                int            vocab_size_padded,
                                size_t         step,
                                cudaStream_t   stream);
#endif
template void invokeBanBadWords(float*       logits,
                                const int*   output_ids_buf,
                                const int*   parent_ids_buf,
                                int          batch_size,
                                int          local_batch_size,
                                int          beam_width,
                                const int*   bad_words,
                                bool         share_words,
                                size_t       bad_words_len,
                                int          id_offset,
                                int          vocab_size_padded,
                                size_t       step,
                                cudaStream_t stream);






template<typename T>
__global__ void retain_option_last_tokens_stage_1(  T*           logits,                // [beam_width, vocab_size_padded]
                                                    const int*   option_last_ids,       // [max_option_last_count]
                                                    int*         is_option_last_token,  // [vocab_size_padded]
                                                    int          beam_width,
                                                    size_t       max_option_last_count,
                                                    int          vocab_size_padded)
{
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int stride = stride_x * stride_y;
    int idx = idx_y * stride_x + idx_x;

    for (int i = idx; i < vocab_size_padded; i += stride) {
        is_option_last_token[i] = 0;
    }
}

template<typename T>
__global__ void retain_option_last_tokens_stage_2(  T*           logits,                // [beam_width, vocab_size_padded]
                                                    const int*   option_last_ids,       // [max_option_last_count]
                                                    int*         is_option_last_token,  // [vocab_size_padded]
                                                    int          beam_width,
                                                    size_t       max_option_last_count,
                                                    int          vocab_size_padded)
{
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int stride = stride_x * stride_y;
    int idx = idx_y * stride_x + idx_x;

    for (int i = idx; i < max_option_last_count; i += stride) {
        is_option_last_token[option_last_ids[i]] = 1;
    }
}

template<typename T>
__global__ void retain_option_last_tokens_stage_3(  T*           logits,                // [beam_width, vocab_size_padded]
                                                    const int*   option_last_ids,       // [max_option_last_count]
                                                    int*         is_option_last_token,  // [vocab_size_padded]
                                                    int          beam_width,
                                                    size_t       max_option_last_count,
                                                    int          vocab_size_padded)
{
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    for (int i = idx_x; i < vocab_size_padded; i += stride_x) {
        for (int j = idx_y; j < beam_width; j += stride_y) {
            if (is_option_last_token[i] < 0.5) {
                logits[j * vocab_size_padded + i] = static_cast<T>(-INFINITY);
            }
        }
    }
}


template<typename T>
void invokeRetainOptionLastTokens(  T*              logits,                 // [beam_width, vocab_size_padded]
                                    const int*      option_last_ids,        // [max_option_last_count]
                                    int*            is_option_last_token,   // [vocab_size_padded]
                                    int             beam_width,
                                    size_t          max_option_last_count,
                                    int             vocab_size_padded,
                                    cudaStream_t    stream)
{
    dim3 block, grid;
    block.x = 256;
    grid.x = (vocab_size_padded + 2048 - 1) / 2048;
    grid.y = beam_width;

    retain_option_last_tokens_stage_1<<<grid, block, 0, stream>>>(  logits, 
                                                                    option_last_ids, 
                                                                    is_option_last_token, 
                                                                    beam_width, 
                                                                    max_option_last_count, 
                                                                    vocab_size_padded);

    retain_option_last_tokens_stage_2<<<grid, block, 0, stream>>>(  logits, 
                                                                    option_last_ids, 
                                                                    is_option_last_token, 
                                                                    beam_width, 
                                                                    max_option_last_count, 
                                                                    vocab_size_padded);
    
    retain_option_last_tokens_stage_3<<<grid, block, 0, stream>>>(  logits, 
                                                                    option_last_ids, 
                                                                    is_option_last_token, 
                                                                    beam_width, 
                                                                    max_option_last_count, 
                                                                    vocab_size_padded);

    sync_check_cuda_error();
}


template void invokeRetainOptionLastTokens( half*           logits,                 // [beam_width, vocab_size_padded]
                                            const int*      option_last_ids,        // [max_option_last_count]
                                            int*            is_option_last_token,   // [vocab_size_padded]
                                            int             beam_width,
                                            size_t          max_option_last_count,
                                            int             vocab_size_padded,
                                            cudaStream_t    stream);
#ifdef ENABLE_BF16
template void invokeRetainOptionLastTokens( __nv_bfloat16*  logits,                 // [beam_width, vocab_size_padded]
                                            const int*      option_last_ids,        // [max_option_last_count]
                                            int*            is_option_last_token,   // [vocab_size_padded]
                                            int             beam_width,
                                            size_t          max_option_last_count,
                                            int             vocab_size_padded,
                                            cudaStream_t    stream);
#endif
template void invokeRetainOptionLastTokens( float*          logits,                 // [beam_width, vocab_size_padded]
                                            const int*      option_last_ids,        // [max_option_last_count]
                                            int*            is_option_last_token,   // [vocab_size_padded]
                                            int             beam_width,
                                            size_t          max_option_last_count,
                                            int             vocab_size_padded,
                                            cudaStream_t    stream);

}  // namespace fastertransformer
