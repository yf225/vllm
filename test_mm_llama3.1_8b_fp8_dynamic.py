"""
How to collect NCU metrics for mm kernels:
1. Run the llama model with command: `wp <vllm command> >output.txt 2>&1` (make sure vLLM is using our custom FlopCounterMode),
2. Extract the mm kernel shapes from the log file with `python ~/extract_kernel_shapes.py <kernel_name> output.txt >kernel_shapes.csv 2>&1` (optionally only extract top-N-flops kernels),
3. Put the mm kernel shapes into this file (run_mm_operations()) and run the script, to get the combined CSV file.
4. Run `python ~/ncu_trace_parser.py <combined_CSV_file>` to get the NCU metrics summary.
"""

import torch
from collections import defaultdict
import csv
from io import StringIO
from datetime import datetime

def create_tensor_with_strides(shape, stride, dtype, device):
    # Create tensor with specific shape and strides
    tensor = torch.randn(*shape, dtype=dtype, device=device)
    tensor.as_strided(shape, stride)
    return tensor

def create_run_script(out_shape, out_stride, out_type,
                   a_shape, a_stride, a_type, a_scale_shape, a_scale_stride, a_scale_type,
                   b_shape, b_stride, b_type, b_scale_shape, b_scale_stride, b_scale_type,
                   script_path):
    """Create a run script for a specific matrix multiplication shape with scaling"""
    # Convert dtype strings to actual type names
    out_dtype_str = out_type.split('.')[-1]
    a_dtype_str = a_type.split('.')[-1]
    b_dtype_str = b_type.split('.')[-1]
    a_scale_dtype_str = a_scale_type.split('.')[-1]
    b_scale_dtype_str = b_scale_type.split('.')[-1]
    
    script_content = f"""
import torch

def run_mm():
    device = torch.device("cuda")
    # Output and input dtypes
    out_dtype = torch.{out_dtype_str}
    a_dtype = torch.{a_dtype_str}
    b_dtype = torch.{b_dtype_str}
    scale_dtype = torch.{a_scale_dtype_str}  # Both scales use same type (float32)
    
    # Create input tensors with specific shapes and strides
    mat_a = torch.randn({a_shape}, dtype=a_dtype, device=device)
    mat_a = mat_a.as_strided({a_shape}, {a_stride})
    
    mat_b = torch.randn({b_shape}, dtype=b_dtype, device=device)
    mat_b = mat_b.as_strided({b_shape}, {b_stride})
    
    # Create scale tensors
    scale_a = torch.ones({a_scale_shape}, dtype=scale_dtype, device=device)
    scale_a = scale_a.as_strided({a_scale_shape}, {a_scale_stride})
    
    scale_b = torch.ones({b_scale_shape}, dtype=scale_dtype, device=device)
    scale_b = scale_b.as_strided({b_scale_shape}, {b_scale_stride})
    
    # Create output tensor
    out = torch.empty({out_shape}, dtype=out_dtype, device=device)
    out = out.as_strided({out_shape}, {out_stride})
    
    # Warmup
    for _ in range(5):
        torch.ops._C.cutlass_scaled_mm(out, mat_a, mat_b, scale_a, scale_b, None)
    
    # Actual run
    torch.ops._C.cutlass_scaled_mm(out, mat_a, mat_b, scale_a, scale_b, None)
    
if __name__ == "__main__":
    run_mm()
"""
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path

def run_ncu_trace(kernel_id, script_path):
    """Run NCU trace for a specific kernel ID and script"""
    import subprocess
    import os
    from datetime import datetime
    import csv
    from io import StringIO
    
    # Create the full command that sources bashrc and runs the command
    cmd = f"""
source ~/.bashrc
export CUDA_INJECTION64_PATH=none
dyno dcgm_profiling --mute=true --duration=100000_s >/dev/null 2>&1

# Define metrics to collect
METRICS="dram__bytes.sum.per_second,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,group:memory__shared_table"

# Generate the report file with specific metrics (suppress output)
$SUDO ${{CUDA_HOME}}/bin/ncu \\
    --metrics "$METRICS" \\
    --import-source yes \\
    -o "{kernel_id}_profile" \\
    -f \\
    python {script_path} >/dev/null 2>&1

# Convert to CSV with the same metrics (suppress output)
${{CUDA_HOME}}/bin/ncu -i "{kernel_id}_profile.ncu-rep" \\
    --csv \\
    --page raw \\
    --metrics "$METRICS" >"{kernel_id}_metrics.csv" 2>/dev/null

# Only show the CSV contents
cat "{kernel_id}_metrics.csv"
rm -rf "{kernel_id}_profile.ncu-rep"
rm -rf "{kernel_id}_metrics.csv"
"""
    
    try:
        # Use bash to execute the commands
        result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error running NCU trace for kernel {kernel_id}: {e}")
        return None
    finally:
        # Cleanup
        if os.path.exists(script_path):
            os.remove(script_path)

def convert_to_gbytes(value, current_unit):
    """Convert value to GBytes/s from either TBytes/s or GBytes/s"""
    value = float(value)
    if "TByte/s".lower() in current_unit.lower():
        return value * 1024  # Convert TBytes to GBytes
    elif "GByte/s".lower() in current_unit.lower():
        return value
    else:
        raise ValueError(f"Unexpected unit {current_unit}")

def run_mm_operations():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Each entry is now a tuple of:
    # ((output_shape, output_stride, output_type,
    #   input_a_shape, input_a_stride, input_a_type, input_a_scale_shape, input_a_scale_stride, input_a_scale_type,
    #   input_b_shape, input_b_stride, input_b_type, input_b_scale_shape, input_b_scale_stride, input_b_scale_type), flop%)
    shapes_with_percentages = [
        # 50.08% of FLOPs
        (((64, 28672), (28672, 1), 'torch.bfloat16',
          (64, 4096), (4096, 1), 'torch.float8_e4m3fn', (64, 1), (1, 1), 'torch.float32',
          (4096, 28672), (1, 4096), 'torch.float8_e4m3fn', (28672, 1), (1, 1), 'torch.float32'), 50.08),
        
        # 25.04% of FLOPs
        (((64, 4096), (4096, 1), 'torch.bfloat16',
          (64, 14336), (14336, 1), 'torch.float8_e4m3fn', (64, 1), (1, 1), 'torch.float32',
          (14336, 4096), (1, 14336), 'torch.float8_e4m3fn', (4096, 1), (1, 1), 'torch.float32'), 25.04),
        
        # 10.73% of FLOPs
        (((64, 6144), (6144, 1), 'torch.bfloat16',
          (64, 4096), (4096, 1), 'torch.float8_e4m3fn', (64, 1), (1, 1), 'torch.float32',
          (4096, 6144), (1, 4096), 'torch.float8_e4m3fn', (6144, 1), (1, 1), 'torch.float32'), 10.73),
        
        # 7.15% of FLOPs
        (((64, 4096), (4096, 1), 'torch.bfloat16',
          (64, 4096), (4096, 1), 'torch.float8_e4m3fn', (64, 1), (1, 1), 'torch.float32',
          (4096, 4096), (1, 4096), 'torch.float8_e4m3fn', (4096, 1), (1, 1), 'torch.float32'), 7.15),
    ]
    
    results = []
    all_csv_data = []
    header = None
    unit_row = None
    
    for idx, ((out_shape, out_stride, out_type,
               a_shape, a_stride, a_type, a_scale_shape, a_scale_stride, a_scale_type,
               b_shape, b_stride, b_type, b_scale_shape, b_scale_stride, b_scale_type), percentage) in enumerate(shapes_with_percentages):
        kernel_id = f"mm_kernel_{idx}"
        script_path = f"run_mm_{idx}.py"
        
        # Create run script for this shape with new parameters including scales
        create_run_script(out_shape, out_stride, out_type,
                         a_shape, a_stride, a_type, a_scale_shape, a_scale_stride, a_scale_type,
                         b_shape, b_stride, b_type, b_scale_shape, b_scale_stride, b_scale_type,
                         script_path)
        
        # Run NCU trace and collect CSV output
        csv_output = run_ncu_trace(kernel_id, script_path)
        if csv_output:
            csv_reader = csv.reader(StringIO(csv_output))
            csv_data = list(csv_reader)
            assert len(csv_data) >= 3
            
            current_header = csv_data[0]
            current_units = csv_data[1]
            
            # Store header from first CSV
            if header is None:
                header = current_header
                unit_row = current_units[:]
                # Force DRAM bandwidth unit to GByte/s
                dram_col_idx = header.index("dram__bytes.sum.per_second")
                unit_row[dram_col_idx] = "GByte/s"
                # Add FLOP% column
                header.append("FLOP%")
                unit_row.append("%")
                all_csv_data.extend([header, unit_row])
            
            # Convert and add data rows
            for row in csv_data[2:]:
                dram_col_idx = header.index("dram__bytes.sum.per_second")
                # Convert the DRAM bandwidth value to GBytes/s
                row[dram_col_idx] = f"{convert_to_gbytes(row[dram_col_idx], current_units[dram_col_idx]):.2f}"
                row.append(f"{percentage:.2f}")  # Add FLOP%
                all_csv_data.append(row)
        
        try:
            # Create tensors with specific strides
            mat1 = create_tensor_with_strides(out_shape, out_stride, eval(out_type), device)
            mat2 = create_tensor_with_strides(a_shape, a_stride, eval(a_type), device)
            result = torch.mm(mat1, mat2)
            results.append(result)
            print(f"Successfully computed mm for shapes {out_shape} x {a_shape} -> {result.shape} (Combined {percentage:.2f}% of FLOPs)")
        except RuntimeError as e:
            print(f"Failed for shapes {out_shape} x {a_shape} (Combined {percentage:.2f}% of FLOPs): {e}")

    # Save combined CSV data to file with timestamp
    timestamp = int(datetime.now().timestamp())
    output_file = f"{timestamp}_metrics.csv"
    with open(output_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(all_csv_data)

    # Print summary of deduplication
    print("\nShape deduplication summary:")
    print(f"# of shapes: {len(shapes_with_percentages)}")

    print(f"\nSaved combined metrics to: {output_file}")

if __name__ == "__main__":
    run_mm_operations() 
