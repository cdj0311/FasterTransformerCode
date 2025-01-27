/*
 * Copyright (c) 2022-2022, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <string>
#include <unordered_map>

#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/DynamicDecodeBaseLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopPSamplingLayer.h"

namespace fastertransformer {

template<typename T>
class DynamicDecodeLayer: public BaseLayer {
protected:
    void allocateBuffer() override;
    void freeBuffer() override;
    void initialize();
    bool hasDiffRuntimeArgs(const std::unordered_map<std::string, Tensor>* input_tensors);

    DynamicDecodeBaseLayer* online_beamsearch_decode_;
    DynamicDecodeBaseLayer* beamsearch_decode_;
    DynamicDecodeBaseLayer* topk_decode_;
    DynamicDecodeBaseLayer* topp_decode_;

    size_t          vocab_size_;
    size_t          vocab_size_padded_;
    cudaDeviceProp* cuda_device_prop_;

    // List of argument names which can have different values in runtime
    // and does not support a batched version of kernel in beam search.
    const std::vector<std::string> runtime_arg_names_ = {
        "beam_search_diversity_rate", "temperature", "len_penalty", "repetition_penalty"};
    bool has_diff_runtime_args_ = false;
    int* finished_sum_          = nullptr;
    int* is_option_last_token_  = nullptr;

public:
    DynamicDecodeLayer(size_t           vocab_size,
                       size_t           vocab_size_padded,
                       int              end_id,
                       cudaStream_t     stream,
                       cublasMMWrapper* cublas_wrapper,
                       IAllocator*      allocator,
                       bool             is_free_buffer_after_forward,
                       cudaDeviceProp*  cuda_device_prop);

    ~DynamicDecodeLayer();
    DynamicDecodeLayer(DynamicDecodeLayer const& dynamic_decode_layer);

    void setup(const size_t                                   batch_size,
               const size_t                                   beam_width,
               const std::unordered_map<std::string, Tensor>* runtime_args);
    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors);
};

}  // namespace fastertransformer
