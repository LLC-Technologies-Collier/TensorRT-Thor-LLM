import os
import shutil
import torch
# FORCE FLOAT16 GLOBALLY to try and fix the 2GB ONNX limit
torch.set_default_dtype(torch.float16)

from tensorrt_edgellm.quantization.llm_quantization import quantize_and_save_llm
from tensorrt_edgellm.onnx_export.llm_export import export_llm_model

# --- CONFIGURATION ---
RAW_MODEL_PATH = os.path.expanduser("~/jax_workspace/gemma_download_raw")
QUANTIZED_CKPT_PATH = os.path.expanduser("~/gemma3_nvfp4_ckpt")
ONNX_OUTPUT_PATH = os.path.expanduser("~/gemma3_onnx")

def main():
#     # ---------------------------------------------------------
#     # STEP 1: Quantize (Float/BFloat16 -> NVFP4)
#     # ---------------------------------------------------------
#     # print(f"\n=== STEP 1: Quantizing Model to NVFP4 (Thor Native) ===")
    
#     if os.path.exists(QUANTIZED_CKPT_PATH):
#         print(f"Found existing checkpoint at {QUANTIZED_CKPT_PATH}. Skipping Step 1.")
#     else:
#         print("Starting quantization... (This downloads 'cnn_dailymail' for calibration)")
#         try:
#             quantize_and_save_llm(
#                 model_dir=RAW_MODEL_PATH,
#                 output_dir=QUANTIZED_CKPT_PATH,
#                 quantization="nvfp4",
# #                dtype="float16",
# #                dtype="fp16",
#                 dtype="bf16",
#                 dataset_dir="cnn_dailymail" 
#             )
#         except Exception as e:
#             print(f"\n!!! QUANTIZATION FAILED !!!")
#             print(f"Error: {e}")
#             return

    # ---------------------------------------------------------
    # STEP 2: Export (NVFP4 Checkpoint -> ONNX)
    # ---------------------------------------------------------
    print(f"\n=== STEP 2: Exporting to ONNX ===")

    try:
        # We removed the invalid 'dtype' argument.
        # We rely on torch.set_default_dtype(torch.float16) at the top 
        # to hopefully shrink the file size.
        export_llm_model(
            model_dir=QUANTIZED_CKPT_PATH, 
            output_dir=ONNX_OUTPUT_PATH,
            device="cuda"
        )
        print(f"\n=== SUCCESS ===")
        print(f"Your Thor-optimized ONNX files are ready in: {ONNX_OUTPUT_PATH}")
      
    except Exception as e:
        print(f"\n!!! EXPORT FAILED !!!")
        print(f"Error: {e}")
        # We exit with code 1 so the bash script knows to stop
        import sys
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 3: Build TensorRT Engine (ONNX -> Engine)
    # ---------------------------------------------------------
    print(f"\n=== STEP 3: Building TensorRT Engine ===")
    
    ENGINE_OUTPUT_PATH = os.path.expanduser("~/gemma3_engine")
    
    # Define the build command
    # --gemm_plugin: Accelerates matrix multiplication
    # --max_batch_size: 1 is safest for 128GB Unified Memory
    # --max_input_len / --max_output_len: Standard chat context windows
    build_cmd = (
        f"trtllm-build "
        f"--checkpoint_dir {ONNX_OUTPUT_PATH} "
        f"--output_dir {ENGINE_OUTPUT_PATH} "
        f"--gemm_plugin float16 "  # Use float16 plugin for speed
        f"--max_batch_size 1 "
        f"--max_input_len 2048 "
        f"--max_output_len 512 "
        f"--workers 14 "           # Use all your CPU cores!
    )

    print(f"Running build command: {build_cmd}")
    
    try:
        # Execute the shell command
        exit_code = os.system(build_cmd)
        
        if exit_code != 0:
            raise Exception(f"Build command failed with code {exit_code}")
            
        print(f"\n=== VICTORY ===")
        print(f"Engine built successfully in: {ENGINE_OUTPUT_PATH}")
        print(f"You can now run the server!")
        
    except Exception as e:
        print(f"\n!!! BUILD FAILED !!!\nError: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
