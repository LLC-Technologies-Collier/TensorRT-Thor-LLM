# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import time
import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from modelopt.onnx.llm_export_utils.surgeon_utils import fold_fp8_qdq_to_dq
from modelopt.onnx.quantization.qdq_utils import (fp4qdq_to_2dq,
                                                  quantize_weights_to_int4,
                                                  quantize_weights_to_mxfp8)
# Import external data helper
from onnx.external_data_helper import convert_model_to_external_data

from ..common import ONNX_OPSET_VERSION
from ..llm_models.layers.int4_gemm_plugin import int4_dq_gemm_to_plugin
from ..llm_models.models.llm_model import EdgeLLMModelForCausalLM

def is_int4_awq_quantized(model: nn.Module) -> bool:
    for _, module in model.named_modules():
        if (hasattr(module, "input_quantizer")
                and hasattr(module, "weight_quantizer")
                and module.weight_quantizer._num_bits == 4
                and module.input_quantizer._disabled):
            return True
    return False

def is_fp4_quantized(model: nn.Module) -> bool:
    for _, module in model.named_modules():
        if (hasattr(module, "input_quantizer")
                and module.input_quantizer.block_sizes
                and module.input_quantizer.block_sizes.get("scale_bits", None) == (4, 3)):
            return True
    return False

def is_mxfp8_quantized(model: nn.Module) -> bool:
    for _, module in model.named_modules():
        if (hasattr(module, "input_quantizer")
                and module.input_quantizer.block_sizes
                and module.input_quantizer.block_sizes.get("scale_bits", None) == (8, 0)):
            return True
    return False

def is_fp8_quantized(model: nn.Module) -> bool:
    for _, module in model.named_modules():
        if (hasattr(module, "input_quantizer")
                and module.input_quantizer._num_bits == (4, 3)
                and hasattr(module, "weight_quantizer")
                and module.weight_quantizer._num_bits == (4, 3)):
            return True
    return False

def untie_nvfp4_lm_head_initializer(model: onnx.ModelProto) -> onnx.ModelProto:
    LM_HEAD_WEIGHT_NAME = "lm_head.weight"
    EMBED_TOKENS_WEIGHT_NAME = "embed_tokens.weight"
    lmhead_weight_quantizer = None
    for node in model.graph.node:
        if node.name == "/lm_head/weight_quantizer/TRT_FP4QDQ":
            lmhead_weight_quantizer = node
            break
    if lmhead_weight_quantizer is None:
        return model # Safely return if node not found

    if lmhead_weight_quantizer.input and EMBED_TOKENS_WEIGHT_NAME in lmhead_weight_quantizer.input[0]:
        embed_init = None
        for init in model.graph.initializer:
            if EMBED_TOKENS_WEIGHT_NAME in init.name:
                embed_init = init
                break
        if embed_init is None:
            return model

        print(f"Untying lm_head weights from {lmhead_weight_quantizer.input[0]}")
        new_init = copy.deepcopy(embed_init)
        new_init.name = LM_HEAD_WEIGHT_NAME
        model.graph.initializer.append(new_init)
        lmhead_weight_quantizer.input[0] = LM_HEAD_WEIGHT_NAME

    return model

def fix_model_int4_output_dtypes(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    graph = onnx_model.graph
    output_to_node = {}
    for node in graph.node:
        for output in node.output:
            output_to_node[output] = node
    graph_outputs = {output.name: output for output in graph.output}

    def set_cast_dtype(node, dtype):
        for attr in node.attribute:
            if attr.name == "to":
                attr.i = dtype
                return

    if "logits" in graph_outputs:
        logits = graph_outputs["logits"]
        producer = output_to_node.get(logits.name)
        if producer and producer.op_type == "LogSoftmax":
            cast_node = output_to_node.get(producer.input[0])
            if cast_node and cast_node.op_type == "Cast":
                set_cast_dtype(cast_node, 1)
        elif producer and producer.op_type == "Cast":
            set_cast_dtype(producer, 1)
        logits.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    if "hidden_states" in graph_outputs:
        hidden_states = graph_outputs["hidden_states"]
        producer = output_to_node.get(hidden_states.name)
        if hidden_states.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16:
            if producer and producer.op_type == "Cast":
                set_cast_dtype(producer, 10)
            else:
                intermediate = f"{hidden_states.name}_pre_fp16"
                if producer:
                    for i, out in enumerate(producer.output):
                        if out == hidden_states.name:
                            producer.output[i] = intermediate
                cast = onnx.helper.make_node("Cast", inputs=[intermediate], outputs=[hidden_states.name], to=10, name=f"{hidden_states.name}_cast_fp16")
                graph.node.append(cast)
            hidden_states.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16

    return onnx_model

def export_onnx(model, inputs, output_dir, input_names, output_names, dynamic_axes):
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = f'{output_dir}/model.onnx'
    
    print(f"[PATCH] Exporting to {onnx_path}...")
    with torch.inference_mode():
        torch.onnx.export(model, inputs, onnx_path, export_params=True, 
                          dynamic_axes=dynamic_axes, input_names=input_names, 
                          output_names=output_names, opset_version=ONNX_OPSET_VERSION, 
                          do_constant_folding=True, dynamo=False)
                          
    t1 = time.time()
    print(f"ONNX export completed in {t1 - t0}s. Apply post-processing...")
    
    # PATCH: Load WITH data this time, but handle optimization failures gracefully
    print("[PATCH] Loading ONNX model (Standard Load)...")
    try:
        onnx.shape_inference.infer_shapes_path(onnx_path)
        # We try standard load first to satisfy the optimizer
        onnx_model = onnx.load(onnx_path, load_external_data=True)
    except Exception as e:
        print(f"!!! Warning: Standard load failed ({e}). Trying lazy load...")
        onnx_model = onnx.load(onnx_path, load_external_data=False)
    
    graph = None

    if is_int4_awq_quantized(model):
        print("INT4 AWQ quantization detected...")
        onnx_model = quantize_weights_to_int4(onnx_model)
        onnx_model = fix_model_int4_output_dtypes(onnx_model)
        graph = gs.import_onnx(onnx_model)
        graph = int4_dq_gemm_to_plugin(graph)
        
    if is_fp8_quantized(model):
        print("FP8 quantization detected...")
        if graph is None:
            graph = gs.import_onnx(onnx_model)
        graph = fold_fp8_qdq_to_dq(graph)
        
    if graph is not None:
        onnx_model = gs.export_onnx(graph)

    if isinstance(model, EdgeLLMModelForCausalLM) and is_fp4_quantized(model.lm_head):
        onnx_model = untie_nvfp4_lm_head_initializer(onnx_model)
        
    if is_fp4_quantized(model):
        print("NVFP4 quantization detected...")
        try:
            # THIS IS THE DANGER ZONE. We wrap it.
            onnx_model = fp4qdq_to_2dq(onnx_model)
        except Exception as e:
            print(f"!!! WARNING: Optimization failed: {e}")
            print("!!! Skipping FP4QDQ conversion. Proceeding with raw quantized graph.")
        
    if is_mxfp8_quantized(model):
        print("MXFP8 quantization detected...")
        onnx_model = quantize_weights_to_mxfp8(onnx_model)

    print("Removing temp files...")
    for file in os.listdir(output_dir):
        if file.endswith(".json"): continue
        if file.endswith(".py"): continue
        fp = os.path.join(output_dir, file)
        if os.path.isfile(fp):
            os.remove(fp)

    # PATCH: Save with External Data Forced (Again)
    print(f"[PATCH] Saving final optimized model to {onnx_path}...")
    
    # Force convert to external data to respect 2GB limit
    # This prevents the final save from crashing
    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=False, 
        location="onnx_model.data",
        size_threshold=1024,
        convert_attribute=True
    )
    
    onnx.save_model(onnx_model, onnx_path)
                    
    t2 = time.time()
    print(f"ONNX post-processing completed in {t2 - t1}s. Saved to {output_dir}")
