import torch
import sys

def main():
    print("=================================================")
    print("   GPU Detection")
    print("=================================================")
    print(f"Python version: {sys.version.split(' ')[0]}")
    print(f"PyTorch version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available. GPU is required for this pipeline.")
        sys.exit(1)
        
    gpu_count = torch.cuda.device_count()
    print(f"CUDA is available. Found {gpu_count} GPU(s).")
    
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  Memory: {mem_gb:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        
    print(f"BF16 support: {torch.cuda.is_bf16_supported()}")

if __name__ == "__main__":
    main()
