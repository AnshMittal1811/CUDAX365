# 250DaysStraight
This repo describes my journey with various technologies in the fields of GPU and CUDA, Fundamentals of MHD and Basic HPC tools, CUDA and HPC for MHD, NeRFs, Fluid Simulations, Quantization, Quantum Computing, Quantum Machine Learning, Data Science, Quantum Chromodynamics, LLMs,


Block 1 (Days 1–5)

Day 1:
Set up your WSL2 (Ubuntu) environment for CUDA 12.8.
Verify nvidia-smi and install the latest NVIDIA driver & CUDA toolkit.
Reference: NVIDIA CUDA Setup Guide
```bash
wsl --update

# 2. Open Ubuntu shell, update packages
sudo apt-get update
sudo apt-get upgrade -y

# 3. Add CUDA repository (for 12.8 or whatever the newest release branch is)
#   (See official NVIDIA docs for the exact repo steps if needed)
# Example for Ubuntu 22.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update

# 4. Install CUDA Toolkit (12.8 example)
sudo apt-get install -y cuda-12-8

# 5. Confirm environment variables in ~/.bashrc or equivalent:
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 6. Verify CUDA works
nvidia-smi           # Should show your GPU usage
nvcc --version       # Should show CUDA 12.x

# Download and run samples
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
make -j8
# Then run, e.g.:
./bin/x86_64/linux/release/deviceQuery
./bin/x86_64/linux/release/bandwidthTest

```

Day 2:
Explore simple CUDA samples (deviceQuery, bandwidthTest) and run them.
Familiarize yourself with the structure of a basic CUDA kernel.
Reference: CUDA Samples on GitHub
```bash
sudo apt-get install build-essential cmake

# Download and run samples
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
make -j8
# Then run, e.g.:
./bin/x86_64/linux/release/deviceQuery
./bin/x86_64/linux/release/bandwidthTest
```

Response for DeviceQuery
```bash
(base) anmittal@AnshPredator:/mnt/c/Users/anshm/250DaysStraight/002_Basic_CUDA_Samples/cuda-samples-12.8/Samples/1_Utilities/deviceQuery$ ./build/deviceQuery
./build/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 4090 Laptop GPU"
  CUDA Driver Version / Runtime Version          12.5 / 12.8
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 16376 MBytes (17170956288 bytes)
  (076) Multiprocessors, (128) CUDA Cores/MP:    9728 CUDA Cores
  GPU Max Clock rate:                            1590 MHz (1.59 GHz)
  Memory Clock rate:                             9001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 67108864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.5, CUDA Runtime Version = 12.8, NumDevs = 1
Result = PASS
```

Response for Bandwidth Test
```bash
(base) anmittal@AnshPredator:/mnt/c/Users/anshm/250DaysStraight/002_Basic_CUDA_Samples/cuda-samples-12.8/Samples/1_Utilities/bandwidthTest$ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- The CUDA compiler identification is NVIDIA 12.8.61
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Found CUDAToolkit: /usr/local/cuda/targets/x86_64-linux/include (found version "12.8.61") 
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- Configuring done (39.8s)
-- Generating done (0.0s)
-- Build files have been written to: /mnt/c/Users/anshm/250DaysStraight/002_Basic_CUDA_Samples/cuda-samples-12.8/Samples/1_Utilities/bandwidthTest/build
(base) anmittal@AnshPredator:/mnt/c/Users/anshm/250DaysStraight/002_Basic_CUDA_Samples/cuda-samples-12.8/Samples/1_Utilities/bandwidthTest$ cmake --build build --target  bandwidthTest  -j
[3/3] Linking CUDA executable bandwidthTest
(base) anmittal@AnshPredator:/mnt/c/Users/anshm/250DaysStraight/002_Basic_CUDA_Samples/cuda-samples-12.8/Samples/1_Utilities/bandwidthTest$ ./build/bandwidthTest 
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: NVIDIA GeForce RTX 4090 Laptop GPU
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     12.2

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     9.9

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     1363.1

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```


Day 3:
Work with PTX for Vector Addition and see what happens under the hood:

Commands used: 
```bash
cd ~/cuda-samples-12.8/Samples/0_Introduction/vectorAdd
nvcc -std=c++11 -I ../../../Common \
     -arch=compute_89 -code=compute_89 \
     -O0 -Xptxas -O0 \
     -ptx vectorAdd.cu -o ../../../../../03_vector_add_pts/vectorAdd.ptx

grep -n "ld\.global" vectorAdd.ptx
less vectorAdd.ptx
```

```less

.version 8.7                      // PTX ISA version
.target sm_89                     // target GPU architecture (Ada, RTX 4090 laptop)               
.address_size 64                  // 64-bit pointers

        // .globl       _Z9vectorAddPKfS0_Pfi

.visible .entry _Z9vectorAddPKfS0_Pfi(
        .param .u64 _Z9vectorAddPKfS0_Pfi_param_0,   // A (const float*)
        .param .u64 _Z9vectorAddPKfS0_Pfi_param_1,   // B (const float*)
        .param .u64 _Z9vectorAddPKfS0_Pfi_param_2,   // C (float*)
        .param .u32 _Z9vectorAddPKfS0_Pfi_param_3    // N (int)
)
{
        .reg .pred      %p<2>;     // predicate (boolean) registers
        .reg .f32       %f<5>;     // 32-bit integer regs
        .reg .b32       %r<6>;     // 32-bit integer regs
        .reg .b64       %rd<11>;   // 64-bit integer regs (addresses/offsets)
        ld.param.u64    %rd1, [_Z9vectorAddPKfS0_Pfi_param_0];   // A
        ld.param.u64    %rd2, [_Z9vectorAddPKfS0_Pfi_param_1];   // B
        ld.param.u64    %rd3, [_Z9vectorAddPKfS0_Pfi_param_2];   // C
        ld.param.u32    %r2, [_Z9vectorAddPKfS0_Pfi_param_3];    // N
        mov.u32         %r3, %ntid.x;   // blockDim.x
        mov.u32         %r4, %ctaid.x;  // blockIdx.x
        mov.u32         %r5, %tid.x;    // threadIdx.x
        mad.lo.s32      %r1, %r3, %r4, %r5;   // r1 = blockDim.x*blockIdx.x + threadIdx.x
        setp.ge.s32     %p1, %r1, %r2;        // p1 = (i >= N)
        @%p1 bra        $L__BB0_2;            // if p1, branch to end

        cvta.to.global.u64      %rd4, %rd1;   // convert A from generic to global addr
        mul.wide.s32    %rd5, %r1, 4;         // byte offset = i * sizeof(float)
        add.s64         %rd6, %rd4, %rd5;     // &A[i]
        cvta.to.global.u64      %rd7, %rd2;   // B base
        add.s64         %rd8, %rd7, %rd5;     // &B[i]
        ld.global.f32   %f1, [%rd8];          // f1 = B[i]
        ld.global.f32   %f2, [%rd6];          // f2 = A[i]
        add.f32         %f3, %f2, %f1;        // f3 = A[i] + B[i]
        add.f32         %f4, %f3, 0f00000000; // f4 = f3 + 0.0f (no-op; see note)
        cvta.to.global.u64      %rd9, %rd3;   // C base
        add.s64         %rd10, %rd9, %rd5;    // &C[i]
        st.global.f32   [%rd10], %f4;         // C[i] = f4

$L__BB0_2:
        ret;

}
```
Here, the following description helped me understand what is happening under the hood:
* _Z9vectorAddPKfS0_Pfi is the C++-mangled name of vectorAdd(const float*, const float*, float*, int).
* Kernel args live in a param space; you first load them into registers
* PTX uses virtual registers; the JIT will map them to physical registers
* `%ntid.x`, `%ctaid.x`, `%tid.x` are special registers for the launch geometry.
* `@%p1` is predicate-guarded branch — classic GPU style control flow.
* `cvta.to.global` normalizes a pointer to the global address space (important with unified addressing).
* `mul.wide.s32` does 32-->64-bit multiply for a 64-bit offset.
* That `add.f32 ... + 0.0f` is a harmless compiler artifact (e.g., SSA/value placement or to inhibit certain opt passes). It doesn’t change the result and often disappears with different flags.
* PTX does exactly this: Loads args --> computes i --> checks `i<N` --> computes addresses --> loads `A[i], B[i]` --> adds --> stores to `C[i]`.

For SASS understanding, I did the following:
```bash
mkdir -p "$OUTDIR"
nvcc -std=c++11 -I "$INC" \
     -arch=compute_89 -code=compute_89 \
     -O0 -Xptxas -O0 \
     -ptx "$SRC" -o "$OUT"

ls -lh "$OUT"
grep -n "ld\.global" "$OUT" || true
less "$OUT"

nvcc -std=c++11 -I "$INC" -arch=sm_89 -lineinfo -Xptxas -v \
     "$SRC" -o "$OUTDIR/vectorAdd_sm89"

cuobjdump --dump-sass "$OUTDIR/vectorAdd_sm89" > "$OUTDIR/vectorAdd.sass"
sed -n '1,120p' "$OUTDIR/vectorAdd.sass"

nvcc -std=c++11 -I "$INC" -arch=sm_89 -Xptxas -dlcm=ca "$SRC" -o "$OUTDIR/vectorAdd_ca"
nvcc -std=c++11 -I "$INC" -arch=sm_89 -Xptxas -dlcm=cg "$SRC" -o "$OUTDIR/vectorAdd_cg"
cuobjdump --dump-sass "$OUTDIR/vectorAdd_ca" > "$OUTDIR/ca.sass"
cuobjdump --dump-sass "$OUTDIR/vectorAdd_cg" > "$OUTDIR/cg.sass"
```

Day 4:
Add inline PTX FMAD in the kernel; use Nsight Compute to diff performance
```bash
# Baseline build: keep add separate
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v --fmad=false \
     vec_fma.cu -o vec_add_baseline

# FMA build: allow FMA (and we have inline PTX anyway)
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v \
     vec_fma.cu -o vec_add_fma
```

Reponse: 
```bash
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z18add_inline_ptx_fmaPKfS0_Pfi' for 'sm_89'
ptxas info    : Function properties for _Z18add_inline_ptx_fmaPKfS0_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 1.150 ms
ptxas info    : Compiling entry function '_Z12add_baselinePKfS0_Pfi' for 'sm_89'
ptxas info    : Function properties for _Z12add_baselinePKfS0_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 0.716 ms
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z18add_inline_ptx_fmaPKfS0_Pfi' for 'sm_89'
ptxas info    : Function properties for _Z18add_inline_ptx_fmaPKfS0_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 1.111 ms
ptxas info    : Compiling entry function '_Z12add_baselinePKfS0_Pfi' for 'sm_89'
ptxas info    : Function properties for _Z12add_baselinePKfS0_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 0.673 ms
```

From the compiled code, I get the difference in time for basic FMA:
```bash
(base) anmittal@AnshPredator:/mnt/c/Users/anshm/250DaysStraight/004_fma$ ./vec_add_baseline
./vec_add_fma
baseline_add: 248.818695 ms for 100 iters  (~323.65 GB/s)
inline_ptx_fma: 231.200546 ms for 100 iters  (~348.32 GB/s)
C[0]=3.000000 (expect 3.0)
baseline_add: 275.315765 ms for 100 iters  (~292.50 GB/s)
inline_ptx_fma: 252.080032 ms for 100 iters  (~319.46 GB/s)
C[0]=3.000000 (expect 3.0)


>> cuobjdump --dump-sass vec_add_fma > vec_add_fma_ca.sass
>> cuobjdump --dump-sass vec_add_baseline > vec_add_baseline_ca.sass
```


Day 5:
```bash
>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v --extended-lambda \
     ./005_reduction_warpshuffle_sm/reduction.cu -o ./005_reduction_warpshuffle_sm/reduction

# PTX output (note: -ptx is a top-level flag)
>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v --extended-lambda -ptx \
     ./005_reduction_warpshuffle_sm/reduction.cu -o ./005_reduction_warpshuffle_sm/reduction.ptx

# SASS dump
>> cuobjdump --dump-sass ./005_reduction_warpshuffle_sm/reduction > ./005_reduction_warpshuffle_sm/reduction.sass
./005_reduction_warpshuffle_sm/reduction.cu(131): warning #550-D: variable "K_shared" was set but never used
      auto K_shared = [] __attribute__((device)) (const float*, size_t, float*) {};
           ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

./005_reduction_warpshuffle_sm/reduction.cu(132): warning #550-D: variable "K_shuffle" was set but never used
      auto K_shuffle = [] __attribute__((device)) (const float*, size_t, float*) {};
           ^

ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z21reduce_shuffle_kernelILj256EEvPKfmPf' for 'sm_89'
ptxas info    : Function properties for _Z21reduce_shuffle_kernelILj256EEvPKfmPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 1 barriers, 32 bytes smem, 376 bytes cmem[0]
ptxas info    : Compile time = 23.759 ms
ptxas info    : Compiling entry function '_Z13reduce_sharedILj256EEvPKfmPf' for 'sm_89'
ptxas info    : Function properties for _Z13reduce_sharedILj256EEvPKfmPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 1 barriers, 1024 bytes smem, 376 bytes cmem[0]
ptxas info    : Compile time = 1.974 ms
./005_reduction_warpshuffle_sm/reduction.cu(131): warning #550-D: variable "K_shared" was set but never used
      auto K_shared = [] __attribute__((device)) (const float*, size_t, float*) {};
           ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

./005_reduction_warpshuffle_sm/reduction.cu(132): warning #550-D: variable "K_shuffle" was set but never used
      auto K_shuffle = [] __attribute__((device)) (const float*, size_t, float*) {};

>> ./005_reduction_warpshuffle_sm/reduction
reduce_shared: 166.450 ms for 200 iters, ~322.54 GB/s
reduce_shuffle: 149.106 ms for 200 iters, ~360.06 GB/s
Check: sum=8589934592.0 (expected 67108864.0)

>> cuobjdump --dump-sass ./005_reduction_warpshuffle_sm/reduction > ./005_reduction_warpshuffle_sm/reduction.sass

>> sed -n '1,200p' ./005_reduction_warpshuffle_sm/reduction.ptx | grep -n -i 'shfl\|bar\.sync\|ld\.global\|st\.global'
52:     ld.global.nc.f32        %f6, [%rd2];
61:     ld.global.nc.f32        %f7, [%rd2+1024];
71:     bar.sync        0;
82:     bar.sync        0;
93:     bar.sync        0;
180:    ld.global.nc.f32        %f11, [%rd2];
189:    ld.global.nc.f32        %f12, [%rd2+1024];

>> sudo env "PATH=$PATH" ncu --set full --kernel-name regex:reduce_shared --target-processes all ./005_reduction_warpshuffle_sm/reduction # Need to look at this
```

Day 6:
Wrap the shuffle reduction in the CUDA Graph and replay it
```bash
>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v ./006_graph_shuffle/graph_shuffle.cu -o graph_shuffle
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z21reduce_shuffle_kernelILj256EEvPKfmPf' for 'sm_89'
ptxas info    : Function properties for _Z21reduce_shuffle_kernelILj256EEvPKfmPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 1 barriers, 32 bytes smem, 376 bytes cmem[0]
ptxas info    : Compile time = 38.408 ms
ptxas info    : Compiling entry function '_Z11fill_kernelPfmf' for 'sm_89'
ptxas info    : Function properties for _Z11fill_kernelPfmf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 1.005 ms

>> ./graph_shuffle 
[Direct]   time=398.836 ms, loops=20000, sum=8589934592.0 (expect 4194304.0)
[Graph]    time=313.553 ms, replays=400, kernels/replay=50, total kernels=20000
           sum=8589934592.0 (expect 1677721600.0)
Throughput lower-bound: Direct ~841.31 GB/s  |  Graph ~1070.14 GB/s

>> nvcc -arch=sm_89 -ptx graph_shuffle.cu -o ./graph_shuffle.ptx
>> grep -n "shfl\.sync.down" graph_shuffle.ptx
94:     shfl.sync.down.b32      %r10|%p3, %r5, %r8, %r7, %r9;
99:     shfl.sync.down.b32      %r13|%p4, %r11, %r12, %r7, %r9;
104:    shfl.sync.down.b32      %r16|%p5, %r14, %r15, %r7, %r9;
108:    shfl.sync.down.b32      %r18|%p6, %r17, %r6, %r7, %r9;
113:    shfl.sync.down.b32      %r21|%p7, %r19, %r20, %r7, %r9;
146:    shfl.sync.down.b32      %r35|%p11, %r30, %r33, %r32, %r34;
151:    shfl.sync.down.b32      %r38|%p12, %r36, %r37, %r32, %r34;
156:    shfl.sync.down.b32      %r41|%p13, %r39, %r40, %r32, %r34;
160:    shfl.sync.down.b32      %r43|%p14, %r42, %r31, %r32, %r34;
165:    shfl.sync.down.b32      %r46|%p15, %r44, %r45, %r32, %r34;
>> nsys profile -t cuda,nvtx -o graph_profile ./graph_shuffle
Collecting data...
[Direct]   time=400.533 ms, loops=20000, sum=8589934592.0 (expect 4194304.0)
[Graph]    time=335.940 ms, replays=400, kernels/replay=50, total kernels=20000
           sum=8589934592.0 (expect 1677721600.0)
Throughput lower-bound: Direct ~837.74 GB/s  |  Graph ~998.82 GB/s
Generating '/tmp/nsys-report-f92d.qdstrm'
[1/1] [========================100%] graph_profile.nsys-rep
Generated:
    /mnt/c/Users/anshm/250DaysStraight/006_graph_shuffle/graph_profile.nsys-rep
>> nsys stats --report cudaapisum,gpukernsum --format text graph_profile.nsys-rep
>> nsys export --sqlite true -o graph_profile_sql graph_profile.nsys-rep 
>> nsys profile -t cuda,nvtx,osrt -o graph_profile --force-overwrite=true ./graph_shuffle
Collecting data...
[Direct]   time=466.657 ms, loops=20000, sum=8589934592.0 (expect 4194304.0)
[Graph]    time=348.407 ms, replays=400, kernels/replay=50, total kernels=20000
           sum=8589934592.0 (expect 1677721600.0)
Throughput lower-bound: Direct ~719.04 GB/s  |  Graph ~963.08 GB/s
Generating '/tmp/nsys-report-83f0.qdstrm'
[1/1] [========================100%] graph_profile.nsys-rep
Generated:
    /mnt/c/Users/anshm/250DaysStraight/006_graph_shuffle/graph_profile.nsys-rep
```

Day 7:
PTX Cache and changing streaming editor
```bash
>>> nvcc -std=c++14 -O3 -arch=compute_89 -ptx vec_kernel.cu -o vec_kernel.ptx
>>> grep -n "ld\.global" vec_kernel.ptx | head
44:     ld.global.nc.f32        %f1, [%rd8];
45:     ld.global.nc.f32        %f2, [%rd6];
>>> nvcc -O3 -arch=sm_89 main_vec.cu vec_kernel.ptx -o run_vec_baseline
nvcc fatal   : .ptx input files are only allowed with '--cubin (-cubin)', '--fatbin (-fatbin)', '--device-c (-dc)', '--device-link (-dlink)', '--cubin (-cubin) --device-link (-dlink)', or '--fatbin (-fatbin) --device-link (-dlink)'
>>> ptxas -arch=sm_89 -v vec_kernel.ptx -o vec_kernel_sm89.cubin
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'vec_add_kernel' for 'sm_89'
ptxas info    : Function properties for vec_add_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 32.098 ms
>>> nvcc -O3 -arch=sm_89 main_vec.cu vec_kernel_sm89.cubin -o run_vec_baseline
nvcc fatal   : .cubin input files are only allowed with '--fatbin (-fatbin)', '--device-c (-dc)', '--device-link (-dlink)', '--cubin (-cubin) --device-link (-dlink)', or '--fatbin (-fatbin) --device-link (-dlink)'
>>> nvcc -O3 -arch=sm_89 --cubin main_vec.cu vec_kernel_sm89.cubin -o run_vec_baseline
nvcc fatal   : .cubin input files are only allowed with '--fatbin (-fatbin)', '--device-c (-dc)', '--device-link (-dlink)', '--cubin (-cubin) --device-link (-dlink)', or '--fatbin (-fatbin) --device-link (-dlink)'
>>> nvcc -O3 -arch=sm_89 main_vec.cu --cubin vec_kernel_sm89.cubin -o run_vec_baseline
nvcc fatal   : .cubin input files are only allowed with '--fatbin (-fatbin)', '--device-c (-dc)', '--device-link (-dlink)', '--cubin (-cubin) --device-link (-dlink)', or '--fatbin (-fatbin) --device-link (-dlink)'
>>> nvcc -O3 -arch=sm_89 main_vec.cu vec_kernel.ptx -o run_vec_baseline
nvcc fatal   : .ptx input files are only allowed with '--cubin (-cubin)', '--fatbin (-fatbin)', '--device-c (-dc)', '--device-link (-dlink)', '--cubin (-cubin) --device-link (-dlink)', or '--fatbin (-fatbin) --device-link (-dlink)'
>>> nvcc -O3 -arch=sm_89 main_vec.cu -lcuda -o run_vec
/usr/bin/ld: /tmp/tmpxft_000a061a_00000000-11_main_vec.o: in function 'run_once(int, float*, float*, float*)':
tmpxft_000a061a_00000000-6_main_vec.cudafe1.cpp:(.text+0xaa): undefined reference to `vec_add_kernel'
/usr/bin/ld: tmpxft_000a061a_00000000-6_main_vec.cudafe1.cpp:(.text+0x132): undefined reference to 'vec_add_kernel'
collect2: error: ld returned 1 exit status
>>> ptxas -arch=sm_89 -v vec_kernel.ptx -o vec_kernel_sm89.cubin
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'vec_add_kernel' for 'sm_89'
ptxas info    : Function properties for vec_add_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 2.334 ms
>>> cp vec_kernel.ptx vec_kernel_ca.ptx
>>> sed -i 's/ld\.global\.nc/ld.global.ca/g' vec_kernel_ca.ptx
>>> ptxas -arch=sm_89 -v vec_kernel_ca.ptx -o vec_kernel_ca_sm89.cubin
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'vec_add_kernel' for 'sm_89'
ptxas info    : Function properties for vec_add_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 1.593 ms
>>> grep -n "ld\.global\." vec_kernel.ptx      | head
44:     ld.global.nc.f32        %f1, [%rd8];
45:     ld.global.nc.f32        %f2, [%rd6];
>>> grep -n "ld\.global\." vec_kernel_ca.ptx   | head
44:     ld.global.ca.f32        %f1, [%rd8];
45:     ld.global.ca.f32        %f2, [%rd6];
>>> nvcc -O3 -std=c++17 -arch=sm_89 host_driver.cu -o run_vec_driver -lcuda
>>> ./run_vec_driver vec_kernel_sm89.cubin
C[0]=3.000000 (expect 3.0)
vec_kernel_sm89.cubin : 268.418 ms / 100 iters  (~300.02 GB/s)
>>> ./run_vec_driver vec_kernel_ca_sm89.cubin
C[0]=3.000000 (expect 3.0)
vec_kernel_ca_sm89.cubin : 224.126 ms / 100 iters  (~359.31 GB/s)
>>> cuobjdump --dump-sass vec_kernel_sm89.cubin > sass_base.txt
>>> cuobjdump --dump-sass vec_kernel_ca_sm89.cubin > sass_ca.txt
>>> diff -u sass_base.txt sass_ca.txt | sed -n '1,200p'
--- sass_base.txt       2025-10-21 16:29:14.276404500 -0700
+++ sass_ca.txt 2025-10-21 16:29:14.472837000 -0700
@@ -22,10 +22,10 @@
                                                                                       /* 0x000fc800078e0207 */
         /*0090*/                   IMAD.WIDE R2, R6.reuse, R7.reuse, c[0x0][0x160] ;  /* 0x0000580006027625 */
                                                                                       /* 0x0c0fe400078e0207 */
-        /*00a0*/                   LDG.E.CONSTANT R4, [R4.64] ;                       /* 0x0000000404047981 */
-                                                                                      /* 0x000ea8000c1e9900 */
-        /*00b0*/                   LDG.E.CONSTANT R3, [R2.64] ;                       /* 0x0000000402037981 */
-                                                                                      /* 0x000ea2000c1e9900 */
+        /*00a0*/                   LDG.E.STRONG.SM R4, [R4.64] ;                      /* 0x0000000404047981 */
+                                                                                      /* 0x000ea8000c1eb900 */
+        /*00b0*/                   LDG.E.STRONG.SM R3, [R2.64] ;                      /* 0x0000000402037981 */
+                                                                                      /* 0x000ea2000c1eb900 */
         /*00c0*/                   IMAD.WIDE R6, R6, R7, c[0x0][0x170] ;              /* 0x00005c0006067625 */
                                                                                       /* 0x000fe200078e0207 */
         /*00d0*/                   FADD R9, R4, R3 ;                                  /* 0x0000000304097221 */
>> cuobjdump --dump-sass ./run_vec_driver | sed -n '1,200p'

Fatbin elf code:
================
arch = sm_89
code version = [1,7]
host = linux
compile_size = 64bit

        code for sm_89

Fatbin elf code:
================
arch = sm_89
code version = [1,7]
host = linux
compile_size = 64bit

        code for sm_89

Fatbin ptx code:
================
arch = sm_89
code version = [8,7]
host = linux
compile_size = 64bit
compressed
ptxasOptions = 
```

Day 8:
Warped Matrix Multiplication Accumulate
```bash
>> nvcc -O3 -std=c++17 -arch=sm_89 -Xptxas -v -lineinfo wmma_ptx.cu -o wmma_ptx
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z20wmma_inline_ptx_stubPfi' for 'sm_89'
ptxas info    : Function properties for _Z20wmma_inline_ptx_stubPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 364 bytes cmem[0]
ptxas info    : Compile time = 29.796 ms
ptxas info    : Compiling entry function '_Z15wmma_cxx_kernelPK6__halfS1_Pfiii' for 'sm_89'
ptxas info    : Function properties for _Z15wmma_cxx_kernelPK6__halfS1_Pfiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 6.309 ms
>> ./wmma_ptx
[WMMA C++] C[0,0]=79.000000
[PTX stub] C[0,0]=0.000000 (expect 0 after stub overwrite of first row)
>> cuobjdump --dump-sass ./wmma_ptx | grep -n -E "HMMA|MMA|LDMATRIX|LDSM"
33:        /*0050*/                   HMMA.16816.F32 R4, R4, R8, RZ ;                /* 0x000000080404723c */
278:        /*05d0*/                   HMMA.16816.F32 R20, R8, R24, R20 ;             /* 0x000000180814723c */
286:        /*0610*/                   HMMA.16816.F32 R16, R8, R6, R16 ;              /* 0x000000060810723c */
312:        /*06e0*/                   HMMA.16816.F32 R20, R12, R26, R20 ;            /* 0x0000001a0c14723c */
316:        /*0700*/                   HMMA.16816.F32 R12, R12, R4, R16 ;             /* 0x000000040c0c723c */
344:        /*07e0*/                   HMMA.16816.F32 R20, R8.reuse, R24, R20 ;       /* 0x000000180814723c */
346:        /*07f0*/                   HMMA.16816.F32 R12, R8, R6, R12 ;              /* 0x00000006080c723c */
350:        /*0810*/                   HMMA.16816.F32 R20, R16.reuse, R26, R20 ;      /* 0x0000001a1014723c */
352:        /*0820*/                   HMMA.16816.F32 R16, R16, R4, R12 ;             /* 0x000000041010723c */
444:        /*0b00*/                   HMMA.16816.F32 R20, R4.reuse, R26, R20 ;       /* 0x0000001a0414723c */
446:        /*0b10*/                   HMMA.16816.F32 R16, R4, R28, R16 ;             /* 0x0000001c0410723c */
>> nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo -ptx wmma_ptx.cu -o wmma_ptx.ptx
>> grep -n "mma.sync" wmma_ptx.ptx | sed -n '1,5p'
98:     wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%f82, %f83, %f84, %f85, %f86, %f87, %f88, %f89}, {%r29, %r30, %r31, %r32, %r33, %r34, %r35, %r36}, {%r37, %r38, %r39, %r40, %r41, %r42, %r43, %r44}, {%f145, %f144, %f143, %f142, %f141, %f140, %f139, %f138};
109:    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%f90, %f91, %f92, %f93, %f94, %f95, %f96, %f97}, {%r46, %r47, %r48, %r49, %r50, %r51, %r52, %r53}, {%r54, %r55, %r56, %r57, %r58, %r59, %r60, %r61}, {%f82, %f83, %f84, %f85, %f86, %f87, %f88, %f89};
120:    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%f98, %f99, %f100, %f101, %f102, %f103, %f104, %f105}, {%r63, %r64, %r65, %r66, %r67, %r68, %r69, %r70}, {%r71, %r72, %r73, %r74, %r75, %r76, %r77, %r78}, {%f90, %f91, %f92, %f93, %f94, %f95, %f96, %f97};
131:    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%f145, %f144, %f143, %f142, %f141, %f140, %f139, %f138}, {%r80, %r81, %r82, %r83, %r84, %r85, %r86, %r87}, {%r88, %r89, %r90, %r91, %r92, %r93, %r94, %r95}, {%f98, %f99, %f100, %f101, %f102, %f103, %f104, %f105};
161:    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%f145, %f144, %f143, %f142, %f141, %f140, %f139, %f138}, {%r99, %r100, %r101, %r102, %r103, %r104, %r105, %r106}, {%r107, %r108, %r109, %r110, %r111, %r112, %r113, %r114}, {%f145, %f144, %f143, %f142, %f141, %f140, %f139, %f138};
>> ncu --set full --kernel-name ::wmma_cxx_kernel ./wmma_ptx
==PROF== Connected to process 664418 (/mnt/c/Users/anshm/250DaysStraight/008_wmma_ptx/wmma_ptx)
[WMMA C++] C[0,0]=79.000000
[PTX stub] C[0,0]=0.000000 (expect 0 after stub overwrite of first row)
==PROF== Disconnected from process 664418
==WARNING== No kernels were profiled.
Available Kernels:
1. wmma_cxx_kernel
2. wmma_inline_ptx_stub
>> ncu --set full --kernel-name ::wmma_inline_ptx_stub ./wmma_ptx
==PROF== Connected to process 664554 (/mnt/c/Users/anshm/250DaysStraight/008_wmma_ptx/wmma_ptx)
[WMMA C++] C[0,0]=79.000000
[PTX stub] C[0,0]=0.000000 (expect 0 after stub overwrite of first row)
==PROF== Disconnected from process 664554
==WARNING== No kernels were profiled.
Available Kernels:
1. wmma_cxx_kernel
2. wmma_inline_ptx_stub
```

Pre-Work GEMM Kernels (How to write them? Code Along)
```bash
```

Day 9:
Side-by-side cuBLAS vs WMMA benchmark comparison on tiny GEMMs, then profile regs & SM occupancy

```bash 
>> nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo -Xptxas -v gemm_compare.cu -lcublas -o gemm_compare
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z16wmma_gemm_kernelPK6__halfS1_Pfiii' for 'sm_89'
ptxas info    : Function properties for _Z16wmma_gemm_kernelPK6__halfS1_Pfiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 85.172 ms

>> ./gemm_compare
[WMMA] 128x128x128: 1.908736 ms (200 iters)  ~ 0.439 TFLOP/s
[cuBLAS] 128x128x128: 1.753088 ms (200 iters)  ~ 0.479 TFLOP/s
Samples  Cwmma[0]=89.125   Ccublas[0]=94.125

>> ./gemm_compare 64 64 64
[WMMA] 64x64x64: 1.650688 ms (200 iters)  ~ 0.064 TFLOP/s
[cuBLAS] 64x64x64: 5.917600 ms (200 iters)  ~ 0.018 TFLOP/s
Samples  Cwmma[0]=46.250   Ccublas[0]=46.875

>> ./gemm_compare 32 32 32
[WMMA] 32x32x32: 0.994304 ms (200 iters)  ~ 0.013 TFLOP/s
[cuBLAS] 32x32x32: 1.332160 ms (200 iters)  ~ 0.010 TFLOP/s
Samples  Cwmma[0]=24.750   Ccublas[0]=23.375

>> ncu --set full --kernel-name ::wmma_gemm_kernel ./gemm_compare 128 128 128
==PROF== Connected to process 780960 (/mnt/c/Users/anshm/250DaysStraight/009_gemm_compare/gemm_compare)
[WMMA] 128x128x128: 5.109024 ms (200 iters)  ~ 0.164 TFLOP/s
[cuBLAS] 128x128x128: 5.468256 ms (200 iters)  ~ 0.153 TFLOP/s
Samples  Cwmma[0]=89.125   Ccublas[0]=94.125
==PROF== Disconnected from process 780960
==WARNING== No kernels were profiled.
Available Kernels:
1. Kernel2
2. wmma_gemm_kernel

>> ncu --set full \
    --target-processes all \
    --kernel-name ::wmma_gemm_kernel \
    --launch-skip 20 --launch-count 5 \
    ./gemm_compare 128 128 128
==PROF== Connected to process 782066 (/mnt/c/Users/anshm/250DaysStraight/009_gemm_compare/gemm_compare)
[WMMA] 128x128x128: 3.720192 ms (200 iters)  ~ 0.225 TFLOP/s
[cuBLAS] 128x128x128: 6.237184 ms (200 iters)  ~ 0.134 TFLOP/s
Samples  Cwmma[0]=89.125   Ccublas[0]=94.125
==PROF== Disconnected from process 782066
==WARNING== No kernels were profiled.
Available Kernels:
1. Kernel2
2. wmma_gemm_kernel

> ncu --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__sass_registers_per_thread_alloc.avg,\
sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor_op_hmma.sum \
--kernel-name ::wmma_gemm_kernel \
./gemm_compare 128 128 128
==PROF== Connected to process 782240 (/mnt/c/Users/anshm/250DaysStraight/009_gemm_compare/gemm_compare)
[WMMA] 128x128x128: 3.784704 ms (200 iters)  ~ 0.222 TFLOP/s
[cuBLAS] 128x128x128: 3.912704 ms (200 iters)  ~ 0.214 TFLOP/s
Samples  Cwmma[0]=89.125   Ccublas[0]=94.125
==PROF== Disconnected from process 782240
==WARNING== No kernels were profiled.
Available Kernels:
1. Kernel2
2. wmma_gemm_kernel

> cuobjdump --dump-sass ./gemm_compare | grep -i -E "HMMA|MMA|LDMATRIX" -n
21:             Function : _Z16wmma_gemm_kernelPK6__halfS1_Pfiii
203:        /*05a0*/                   HMMA.16816.F32 R20, R4, R12, R20 ;             /* 0x0000000c0414723c */
215:        /*0600*/                   HMMA.16816.F32 R4, R4, R14, R8 ;               /* 0x0000000e0404723c */
257:        /*0750*/                   HMMA.16816.F32 R20, R16.reuse, R24, R20 ;      /* 0x000000181014723c */
263:        /*0780*/                   HMMA.16816.F32 R16, R16, R12, R4 ;             /* 0x0000000c1010723c */
307:        /*08e0*/                   HMMA.16816.F32 R20, R12.reuse, R24, R20 ;      /* 0x000000180c14723c */
309:        /*08f0*/                   HMMA.16816.F32 R16, R12, R28, R16 ;            /* 0x0000001c0c10723c */
313:        /*0910*/                   HMMA.16816.F32 R20, R8.reuse, R6, R20 ;        /* 0x000000060814723c */
315:        /*0920*/                   HMMA.16816.F32 R8, R8, R4, R16 ;               /* 0x000000040808723c */
407:        /*0c00*/                   HMMA.16816.F32 R20, R4.reuse, R14, R20 ;       /* 0x0000000e0414723c */
409:        /*0c10*/                   HMMA.16816.F32 R8, R4, R34, R8 ;               /* 0x000000220408723c */
```


Day 10:
Start a 2-D explicit Magneto-Hydrodynamics (MHD) solver (ρ, u, B fields) in CUDA C

#### Finite Volume + Rusanov Flux in MagnetoHydroDynamics (MHD)

This is a common numerical scheme used to solve the ideal MHD equations. Let's break it down.

##### 1. The Finite Volume (FV) Method

The Finite Volume method is a numerical technique for solving partial differential equations (PDEs), especially **conservation laws**. The ideal MHD equations are a system of conservation laws.

* **Core Idea:** Divide the simulation domain (e.g., a 2D or 3D space) into many small "finite volumes" or "cells".
* **What it Solves:** Instead of tracking the solution at every *point*, the FV method tracks the **cell-average** of the conserved quantities within each cell.
* **How it Updates:** The change in a cell's average value over time is determined by the **flux** of those quantities across the cell's boundaries (or "faces").

The discrete form of the conservation law for a cell $i$ is:

$$
\frac{d\mathbf{U}_i}{dt} = -\frac{1}{V_i} \sum_{j} \mathbf{F}_{ij}^* A_{ij} + \mathbf{S}_i
$$

* $\mathbf{U}_i$: The vector of cell-averaged **conserved variables** in cell $i$.
* $V_i$: The volume of cell $i$.
* $A_{ij}$: The area of the face between cell $i$ and cell $j$.
* $\mathbf{S}_i$: Source terms (e.g., gravity, resistivity).
* $\mathbf{F}_{ij}^*$: This is the **numerical flux**. It's an approximation of the *true* physical flux across the interface.

The entire challenge of the FV method boils down to finding a good, stable formula for this numerical flux, $\mathbf{F}_{ij}^*$. This is where the **Rusanov flux** comes in.

##### 2. The MHD Equations

In MHD, the vector of conserved variables $\mathbf{U}$ (in 3D) is typically:

$$
\mathbf{U} = \begin{pmatrix}
\rho \\
\rho v_x \\
\rho v_y \\
\rho v_z \\
E \\
B_x \\
B_y \\
B_z
\end{pmatrix}
$$

* $\rho$: Mass density
* $\rho \mathbf{v}$: Momentum (3 components)
* $E$: Total energy density
* $\mathbf{B}$: Magnetic field (3 components)

The flux tensor $\mathbf{F}$ is much more complex, involving fluid pressure $p$, magnetic pressure $B^2/2$, and Maxwell's stress tensor. Solving the "Riemann problem" (what happens when two different states $\mathbf{U}_L$ and $\mathbf{U}_R$ meet at an interface) for MHD is very difficult because it involves 7 or 8 different waves (fast/slow magnetosonic, Alfvén, entropy, and contact waves).

##### 3. The Rusanov (or Lax-Friedrichs) Flux

The Rusanov flux (also known as the local Lax-Friedrichs flux) is a simple, robust, but very **dissipative** (or "diffusive") way to calculate the numerical flux $\mathbf{F}_{ij}^*$. It avoids solving the full, complex MHD Riemann problem.

The formula for the Rusanov flux $\mathbf{F}^*$ between a "Left" state ($\mathbf{U}_L$) and a "Right" state ($\mathbf{U}_R$) at an interface is:

$$
\mathbf{F}^*(\mathbf{U}_L, \mathbf{U}_R) = \frac{1}{2} \left[ \mathbf{F}(\mathbf{U}_L) + \mathbf{F}(\mathbf{U}_R) \right] - \frac{S_{\text{max}}}{2} \left[ \mathbf{U}_R - \mathbf{U}_L \right]
$$

Let's analyze the two parts:

1.  **$\frac{1}{2} \left[ \mathbf{F}(\mathbf{U}_L) + \mathbf{F}(\mathbf{U}_R) \right]$**: This is the "central flux" or the simple average of the physical fluxes from both sides. On its own, this is **unstable** for hyperbolic problems.

2.  **$\frac{S_{\text{max}}}{2} \left[ \mathbf{U}_R - \mathbf{U}_L \right]$**: This is the **numerical dissipation** or "artificial viscosity" term. It's what makes the scheme stable. It "smears" the solution at the interface, preventing non-physical oscillations.

###### The Key Parameter: $S_{\text{max}}$

The crucial part of the Rusanov flux is $S_{\text{max}}$.

* $S_{\text{max}}$ is the **maximum local wave speed** (or maximum eigenvalue of the system) at the interface. It's the fastest speed at which information can travel from either the Left or Right state.
* In MHD, this maximum speed is the **fast magnetosonic wave speed** ($c_f$) plus the normal fluid velocity ($|v_n|$).
* So, $S_{\text{max}}$ is calculated as:
    $$
    S_{\text{max}} = \max \left( |v_{n,L}| + c_{f,L}, |v_{n,R}| + c_{f,R} \right)
    $$
* The fast magnetosonic speed $c_f$ is itself a function of the sound speed $c_s$ and the Alfvén speed $c_A$.

---

##### Summary: Pros and Cons

* **Pro:**
    * **Simple:** Extremely easy to implement. You only need to calculate the physical fluxes $\mathbf{F}(\mathbf{U})$ and the single fastest wave speed $c_f$.
    * **Robust:** Very stable and can handle strong shocks and complex problems without crashing.

* **Con:**
    * **Dissipative:** This is its main drawback. The large amount of numerical diffusion (controlled by $S_{\text{max}}$) smears out sharp features. Shocks, contact discontinuities, and shear waves will look "blurry" compared to solutions from more advanced (and complex) solvers like HLLD, HLL, or Roe.


```bash
> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v mhd2d.cu -o mhd2d
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 38.529 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.490 ms
ptxas info    : Compiling entry function '_Z8step_mhdPKfPfiifff' for 'sm_89'
ptxas info    : Function properties for _Z8step_mhdPKfPfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 72 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 23.984 ms

>./mhd2d
Step 0/200 (dt=4.121e-04)
Step 50/200 (dt=4.051e-04)
Step 100/200 (dt=3.735e-04)
Step 150/200 (dt=3.411e-04)
Center (rho,mx,my,Bx,By,E): 0.815766 0.010215 -0.001882 -0.006862 0.034297 1.917913

> ./mhd2d 128 128 200
Step 0/200 (dt=8.245e-04)
Step 50/200 (dt=7.565e-04)
Step 100/200 (dt=6.320e-04)
Step 150/200 (dt=2.445e-04)
Center (rho,mx,my,Bx,By,E): 0.565025 0.019125 0.004931 -0.043550 0.093441 1.196366

> ./mhd2d 256 256 2000
Step 0/2000 (dt=4.121e-04)
Step 50/2000 (dt=4.051e-04)
Step 100/2000 (dt=3.735e-04)
Step 150/2000 (dt=3.411e-04)
Step 200/2000 (dt=2.390e-04)
Step 250/2000 (dt=7.677e-05)
Step 300/2000 (dt=3.651e-05)
Step 350/2000 (dt=2.413e-05)
Step 400/2000 (dt=1.888e-05)
Step 450/2000 (dt=1.542e-05)
Step 500/2000 (dt=1.268e-05)
Step 550/2000 (dt=1.311e-05)
Step 600/2000 (dt=1.372e-05)
Step 650/2000 (dt=1.521e-05)
Step 700/2000 (dt=1.562e-05)
Step 750/2000 (dt=1.569e-05)
Step 800/2000 (dt=1.428e-05)
Step 850/2000 (dt=1.231e-05)
Step 900/2000 (dt=1.364e-05)
Step 950/2000 (dt=1.259e-05)
Step 1000/2000 (dt=1.179e-05)
Step 1050/2000 (dt=1.097e-05)
Step 1100/2000 (dt=1.064e-05)
Step 1150/2000 (dt=1.011e-05)
Step 1200/2000 (dt=9.864e-06)
Step 1250/2000 (dt=9.545e-06)
Step 1300/2000 (dt=9.115e-06)
Step 1350/2000 (dt=8.400e-06)
Step 1400/2000 (dt=8.228e-06)
Step 1450/2000 (dt=7.944e-06)
Step 1500/2000 (dt=7.975e-06)
Step 1550/2000 (dt=7.812e-06)
Step 1600/2000 (dt=7.415e-06)
Step 1650/2000 (dt=7.514e-06)
Step 1700/2000 (dt=7.829e-06)
Step 1750/2000 (dt=7.656e-06)
Step 1800/2000 (dt=8.111e-06)
Step 1850/2000 (dt=8.435e-06)
Step 1900/2000 (dt=9.224e-06)
Step 1950/2000 (dt=1.008e-05)
Center (rho,mx,my,Bx,By,E): 0.600461 0.009297 0.001848 -0.018487 0.041579 1.260809
```

The advanced code explanation: 
**Analysis of Advanced 2D MHD Solver (mhd2d_adv.cu)**

This CUDA code implements an advanced 2D magnetohydrodynamics (MHD) solver. It builds upon the Finite Volume (FV) method with a Rusanov flux, but adds several critical features to improve accuracy, stability, and performance.

The state vector `U` is a Structure of Arrays (SoA) for 6 components:
`U = [rho, mx, my, Bx, By, E]`
(mass, x-momentum, y-momentum, x-magnetic-field, y-magnetic-field, total energy)

Here are the key advanced features explained:

**1. 2nd-Order MUSCL Reconstruction**

The standard FV method with a Rusanov flux is only **1st-order accurate**. This means it assumes the data in each cell is a flat, constant value. This is numerically very "diffusive" or "blurry," smearing out sharp features like shocks and contact waves.

This code implements a **2nd-order MUSCL** (Monotonic Upstream-centered Scheme for Conservation Laws) scheme to fix this.

* **Core Idea:** Instead of assuming data is constant in a cell, we reconstruct a *linear slope* within each cell. This gives a much better guess for the values at the cell's left and right faces.
* **Primitive Variables:** The reconstruction is done on the **primitive variables** ($\mathbf{P} = [\rho, v_x, v_y, B_x, B_y, p]$) rather than the conserved variables ($\mathbf{U}$). This is more stable and physically accurate, as the conserved variables (like momentum $\rho v_x$) can have sharp jumps even when the primitive velocity $v_x$ is smooth. The code uses `cons_to_prim()` and `prim_to_cons()` to switch between these.
* **Limiter (`minmod`):** A simple linear reconstruction would be 2nd-order but would create new, non-physical oscillations (wiggles) near shocks. A **limiter** is used to "flatten" the slope back to 0 (making it 1st-order) in regions where oscillations might appear. This code uses the `minmod` limiter:
    $$
    s = \text{minmod}(\Delta_L, \Delta_R, \theta \cdot \Delta_C)
    $$
    where $\Delta_L, \Delta_R, \Delta_C$ are the backward, forward, and central differences. The `minmod` function returns the smallest value in magnitude if all arguments have the same sign, and 0 otherwise. This is a very robust (though diffusive) limiter.
* **Implementation:** The main kernel `step_mhd_muscl_powell` gathers a 3-cell stencil (e.g., `PL`, `PC`, `PR`), calls `recon_face_1D_prim()` to compute the limited slopes, and then extrapolates to find the states at both sides of the cell face (e.g., `P_LR_xR` and `P_LR_xL`). These are converted back to `U` and fed into the `rusanov_flux` solver.


**2. Powell 8-Wave Divergence Cleaning**

A fundamental law of physics (one of Maxwell's equations) is that the magnetic field must be divergence-free: $\nabla \cdot \mathbf{B} = 0$. This implies there are no "magnetic monopoles."

In numerical simulations, floating-point errors can cause a small, non-zero $\nabla \cdot \mathbf{B}$ to appear. This error can grow exponentially and make the entire simulation unstable, leading to a crash.

* **Core Idea:** Powell's method adds a non-conservative **source term** to the right-hand-side of the MHD equations. This source term is proportional to the numerically computed $\nabla \cdot \mathbf{B}$.
* **The Source Term:** The term $S_P$ is added to the update:
    $$
    \frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F} = S_P = -(\nabla \cdot \mathbf{B}) \begin{pmatrix} 0 \\ B_x \\ B_y \\ 0 \\ v_x \\ v_y \\ 0 \\ \mathbf{v} \cdot \mathbf{B} \end{pmatrix}
    $$
    (The code uses the 2D version, so $B_z=v_z=0$). This term is designed to advect and damp the $\nabla \cdot \mathbf{B}$ error, preventing it from growing.
* **Implementation:** Inside the `step_mhd_muscl_powell` kernel, after the main FV update (and enabled by `#if USE_POWELL`):
    1.  `divB` is calculated using a simple, 2nd-order central difference.
    2.  The source term is added directly to the updated state `Un`:
        `Un[1] += -dt * divB * Un[3]; // mx -= dt * divB * Bx`
        ...and so on for `my`, `Bx`, `By`, and `E`.

**3. CFL Timestep Calculation from GPU**

The stability of the explicit FV scheme is limited by the **CFL condition**, which states that the timestep `dt` must be small enough that the fastest wave doesn't travel more than one cell width.
$$
dt \le C_{\text{num}} \frac{\Delta x}{S_{\text{max}}}
$$
To calculate `dt`, we must find the **global maximum wave speed** ($S_{\text{max}}$) across the *entire* simulation grid at *every* step.

* **Core Idea:** Instead of copying the entire $N \times N$ grid from the GPU to the CPU to find this one number (which is extremely slow), we perform a **parallel reduction** entirely on the GPU.
* **Implementation:**
    1.  `kernel_maxspeed`: A kernel is launched where each thread (for each cell) computes its *local* max speed: `smax = fmaxf(fabsf(vx)+cf, fabsf(vy)+cf)`.
    2.  `atomicMaxFloatPos(smax)`: Each thread then uses a GPU **atomic operation** (`atomicMax`) to update a *single* global variable `d_max_speed_bits`. This is a "race" where only the largest value seen by any thread will "win" and remain in `d_max_speed_bits`.
    3.  `compute_dt_gpu`: The host function launches the kernel, waits for it to finish (`cudaDeviceSynchronize`), copies the *single* resulting max value back, and computes `dt`. This is much, much faster than a full `cudaMemcpy`.
    4.  The `__float_as_uint` trick (`atomicMaxFloatPos`) is used because atomic operations on floating-point numbers can be tricky, but for positive floats, their integer representation preserves ordering.


**4. Tracking Conservative Invariants**

For a closed (periodic) system with no sources or sinks, the **total mass** ($\sum \rho$) and **total energy** ($\sum E$) in the box must be conserved. Tracking these values is a critical sanity check to ensure the simulation is physically correct and numerically stable.

* **Core Idea:** This is another **parallel reduction** problem, just like the CFL calculation. We need to compute $\sum \mathbf{U}[0]$ and $\sum \mathbf{U}[5]$ over all cells.
* **Implementation:**
    1.  `reduce_mass_energy`: A kernel is launched to sum these values.
    2.  **Shared Memory:** This kernel uses `extern __shared__ double sm[]` as a high-speed, on-chip cache. Threads within a block first sum their partial results into this shared memory array.
    3.  **Block-level Reduction:** The threads in the block perform a fast reduction *within* shared memory.
    4.  **Global Atomics:** The one "master" thread in each block (`threadIdx.x==0`) atomically adds its block's sub-total (`s_mass[0]`) to the global `double* out_mass`.
* **In `main()`:** The initial `M0` and `E0` are stored. At each logging step, `compute_invariants_gpu` is called, and the relative error `(M-M0)/M0` is printed, which should remain very small (e.g., $10^{-10}$).

```bash
> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v mhd2d_adv.cu -o mhd2d_adv
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z18reduce_mass_energyPKfiPdS1_' for 'sm_89'
ptxas info    : Function properties for _Z18reduce_mass_energyPKfiPdS1_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 1 barriers, 384 bytes cmem[0]
ptxas info    : Compile time = 3.200 ms
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 5.504 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.306 ms
ptxas info    : Compiling entry function '_Z21step_mhd_muscl_powellPKfPfiiffff' for 'sm_89'
ptxas info    : Function properties for _Z21step_mhd_muscl_powellPKfPfiiffff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 95 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compile time = 57.066 ms

> ./mhd2d_adv 128 128 200
Init invariants: Mass=1.63840000e+04  Energy=5.73440039e+04
Step   50/ 200  dt=7.639e-09  Mass=1.63852378e+04  dM=7.555e-05  Energy=5.70173830e+05  dE=8.943e+00
Step  100/ 200  dt=7.203e-09  Mass=1.63852359e+04  dM=7.543e-05  Energy=6.64728658e+05  dE=1.059e+01
Step  150/ 200  dt=6.413e-09  Mass=1.63852344e+04  dM=7.534e-05  Energy=7.59321389e+05  dE=1.224e+01
Step  200/ 200  dt=5.274e-09  Mass=1.63852334e+04  dM=7.528e-05  Energy=8.70095850e+05  dE=1.417e+01
Center (rho,mx,my,Bx,By,E): 0.982538 0.034419 -0.008991 0.012728 0.064614 2.407878
```

Day 11:
Avoiding Banking Conflicts by padding tiles in the above equation

```bash
> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v mhd2d_tile.cu -o mhd2d_tile
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z14step_mhd_tiledILi16ELi16ELi1EEvPKfPfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14step_mhd_tiledILi16ELi16ELi1EEvPKfPfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 63 registers, used 1 barriers, 8208 bytes smem, 388 bytes cmem[0]
ptxas info    : Compile time = 31.827 ms
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 5.675 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.301 ms

> ./mhd2d_tile 256 256 150
step 0/150 (dt=4.121e-04)
step 50/150 (dt=4.051e-04)
step 100/150 (dt=3.735e-04)
Center: rho=0.89654 mx=0.01024 my=-0.00387 Bx=-0.00233 By=0.03217 E=2.17966

> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -DPAD_X=1 -DTILE_X=16 -DTILE_Y=16 mhd2d_tile.cu -o mhd2d_tile_pad
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z14step_mhd_tiledILi16ELi16ELi1EEvPKfPfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14step_mhd_tiledILi16ELi16ELi1EEvPKfPfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 63 registers, used 1 barriers, 8208 bytes smem, 388 bytes cmem[0]
ptxas info    : Compile time = 32.115 ms
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 5.526 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.293 ms

> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -DPAD_X=0 -DTILE_X=16 -DTILE_Y=16 mhd2d_tile.cu -o mhd2d_tile_nopad
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z14step_mhd_tiledILi16ELi16ELi0EEvPKfPfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14step_mhd_tiledILi16ELi16ELi0EEvPKfPfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 63 registers, used 1 barriers, 7776 bytes smem, 388 bytes cmem[0]
ptxas info    : Compile time = 33.069 ms
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 5.783 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.322 ms

> ncu --set full \
    --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
smsp__pipe_lsu_mem_shared_op_stalled_backpressure_per_warp_active.avg \
    ./mhd2d_tile_nopad 512 512 50

> ncu --set full \
    --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
smsp__pipe_lsu_mem_shared_op_stalled_backpressure_per_warp_active.avg \
    ./mhd2d_tile_pad 512 512 50

> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v mhd2d_muscl_hll.cu -o mhd_muscl_hll
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z18reduce_mass_energyPKfiPdS1_' for 'sm_89'
ptxas info    : Function properties for _Z18reduce_mass_energyPKfiPdS1_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 1 barriers, 384 bytes cmem[0]
ptxas info    : Compile time = 2.720 ms
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 14.241 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.485 ms
ptxas info    : Compiling entry function '_Z21step_mhd_muscl_powellPKfPfiiffff' for 'sm_89'
ptxas info    : Function properties for _Z21step_mhd_muscl_powellPKfPfiiffff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compile time = 57.800 ms

> ./mhd_muscl_hll 256 256 200
Init invariants: Mass=6.55360000e+04  Energy=2.29376016e+05  (USE_HLL=1, USE_POWELL=1)
Step   50/ 200  dt=2.224e-10  Mass=6.55351947e+04  dM=-1.229e-05  Energy=1.57430925e+07  dE=6.763e+01
Step  100/ 200  dt=2.035e-10  Mass=6.55351936e+04  dM=-1.230e-05  Energy=4.08041729e+07  dE=1.769e+02
Step  150/ 200  dt=1.754e-10  Mass=6.55351924e+04  dM=-1.232e-05  Energy=6.92232569e+07  dE=3.008e+02
Step  200/ 200  dt=1.088e-10  Mass=6.55351913e+04  dM=-1.234e-05  Energy=1.08833112e+08  dE=4.735e+02
Center (rho,mx,my,Bx,By,E): 0.996674 0.014877 -0.008712 0.009267 0.027849 2.480634

> nvcc -O3 -arch=sm_89 -DUSE_HLL=0 -lineinfo -Xptxas -v mhd2d_muscl_hll.cu -o mhd_muscl_rusanov
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z18reduce_mass_energyPKfiPdS1_' for 'sm_89'
ptxas info    : Function properties for _Z18reduce_mass_energyPKfiPdS1_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 1 barriers, 384 bytes cmem[0]
ptxas info    : Compile time = 2.420 ms
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 5.034 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.326 ms
ptxas info    : Compiling entry function '_Z21step_mhd_muscl_powellPKfPfiiffff' for 'sm_89'
ptxas info    : Function properties for _Z21step_mhd_muscl_powellPKfPfiiffff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 95 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compile time = 49.991 ms

> ./mhd_muscl_rusanov 256 256 200
./mhd_muscl_rusanov 256 256 200
Init invariants: Mass=6.55360000e+04  Energy=2.29376016e+05  (USE_HLL=0, USE_POWELL=1)
Step   50/ 200  dt=9.870e-10  Mass=6.55360140e+04  dM=2.131e-07  Energy=1.84444453e+06  dE=7.041e+00
Step  100/ 200  dt=9.158e-10  Mass=6.55360127e+04  dM=1.936e-07  Energy=4.33319315e+06  dE=1.789e+01
Step  150/ 200  dt=2.441e-11  Mass=6.55360117e+04  dM=1.792e-07  Energy=3.96417799e+10  dE=1.728e+05
Step  200/ 200  dt=1.292e-15  Mass=9.30251829e+04  dM=4.195e-01  Energy=2.28094355e+21  dE=9.944e+15
Center (rho,mx,my,Bx,By,E): 0.996392 0.015229 -0.008865 0.009325 0.028031 2.479517
```

TODO: Need to resolve problems with NaNs coming up as the smoothing continues when working with discontinuities due to rho, and plug this kernel 

Day 12:
Add cuFFT for a 2-D Poisson solve step; verify energy spectrum stays consistent

Current issue (why energy/spectrum blew up to inf):
- dt collapses (1e-9 --> 1e-15) once density/pressure approach zero, so max wavespeed spikes and updates inject huge energy.
- FFT spectrum reads inf because fields are already non-finite.
- Likely causes: CFL too aggressive, weak density/pressure floors, sharp rho discontinuities feeding back through the Poisson correction.
- Mitigations to try: lower CFL (e.g., 0.05–0.1), abort if dt < 1e-12 or any field is not finite, tighten rho/pressure floors, use more diffusive flux (HLL) or smaller limiter theta, and log min/max rho/p/E each step to catch the first divergence.
```bash
>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -c step_tiled_poisson.cu -o step_tiled_poisson.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z19power_spectrum_binsPK6float2iiPfPji' for 'sm_89'
ptxas info    : Function properties for _Z19power_spectrum_binsPK6float2iiPfPji
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 2.371 ms
ptxas info    : Compiling entry function '_Z13make_ke_fieldP6float2PKfS2_S2_i' for 'sm_89'
ptxas info    : Function properties for _Z13make_ke_fieldP6float2PKfS2_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 17 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 3.488 ms
ptxas info    : Compiling entry function '_Z14apply_grad_phiPfS_PKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14apply_grad_phiPfS_PKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, used 0 barriers, 396 bytes cmem[0]
ptxas info    : Compile time = 6.133 ms
ptxas info    : Compiling entry function '_Z19complexToRealScaledPfPK6float2if' for 'sm_89'
ptxas info    : Function properties for _Z19complexToRealScaledPfPK6float2if
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 0.546 ms
ptxas info    : Compiling entry function '_Z17copyRealToComplexP6float2PKfi' for 'sm_89'
ptxas info    : Function properties for _Z17copyRealToComplexP6float2PKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.497 ms
ptxas info    : Compiling entry function '_Z13subtract_meanPfif' for 'sm_89'
ptxas info    : Function properties for _Z13subtract_meanPfif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 0.453 ms
ptxas info    : Compiling entry function '_Z21apply_poisson_scalingP6float2PKfii' for 'sm_89'
ptxas info    : Function properties for _Z21apply_poisson_scalingP6float2PKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 2.770 ms
ptxas info    : Compiling entry function '_Z7fill_k2Pfii' for 'sm_89'
ptxas info    : Function properties for _Z7fill_k2Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 368 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compile time = 0.825 ms

>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v mhd2d_adv.cu -lcufft -o mhd_poisson_tiled
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z18reduce_mass_energyPKfiPdS1_' for 'sm_89'
ptxas info    : Function properties for _Z18reduce_mass_energyPKfiPdS1_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 1 barriers, 384 bytes cmem[0]
ptxas info    : Compile time = 2.128 ms
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 4.999 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.263 ms
ptxas info    : Compiling entry function '_Z21step_mhd_muscl_powellPKfPfiiffff' for 'sm_89'
ptxas info    : Function properties for _Z21step_mhd_muscl_powellPKfPfiiffff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 95 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compile time = 45.649 ms
ptxas info    : Compiling entry function '_Z19spectrum_power_binsPK6float2iiPfPji' for 'sm_89'
ptxas info    : Function properties for _Z19spectrum_power_binsPK6float2iiPfPji
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 2.053 ms
ptxas info    : Compiling entry function '_Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i' for 'sm_89'
ptxas info    : Function properties for _Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 17 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 3.316 ms
ptxas info    : Compiling entry function '_Z15complex_to_realPfPK6float2if' for 'sm_89'
ptxas info    : Function properties for _Z15complex_to_realPfPK6float2if
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 0.606 ms
ptxas info    : Compiling entry function '_Z15real_to_complexP6float2PKfi' for 'sm_89'
ptxas info    : Function properties for _Z15real_to_complexP6float2PKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.535 ms
ptxas info    : Compiling entry function '_Z14apply_grad_phiPfS_PKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14apply_grad_phiPfS_PKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, used 0 barriers, 396 bytes cmem[0]
ptxas info    : Compile time = 5.037 ms
ptxas info    : Compiling entry function '_Z13subtract_meanPfif' for 'sm_89'
ptxas info    : Function properties for _Z13subtract_meanPfif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 0.460 ms
ptxas info    : Compiling entry function '_Z21apply_poisson_scalingP6float2PKfii' for 'sm_89'
ptxas info    : Function properties for _Z21apply_poisson_scalingP6float2PKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 2.870 ms
ptxas info    : Compiling entry function '_Z7fill_k2Pfii' for 'sm_89'
ptxas info    : Function properties for _Z7fill_k2Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 368 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compile time = 0.700 ms

>> ./mhd_poisson_tiled 256 256 200
Init invariants: Mass=6.55360000e+04  Energy=2.29376016e+05
Step   50/ 200  dt=9.870e-10  Mass=6.55360140e+04  dM=2.131e-07  Energy=1.84444453e+06  dE=7.041e+00  spectrum[0..7]: 7.934e+10 6.135e+10 6.300e+10 6.141e+10 6.141e+10 6.135e+10 6.123e+10 6.111e+10
Step  100/ 200  dt=9.158e-10  Mass=6.55360127e+04  dM=1.936e-07  Energy=4.33319315e+06  dE=1.789e+01  spectrum[0..7]: 6.788e+10 5.133e+10 5.285e+10 5.139e+10 5.140e+10 5.135e+10 5.126e+10 5.116e+10
Step  150/ 200  dt=2.441e-11  Mass=6.55360117e+04  dM=1.792e-07  Energy=3.96417799e+10  dE=1.728e+05  spectrum[0..7]: 1.571e+21 1.571e+21 1.571e+21 1.571e+21 1.571e+21 1.571e+21 1.571e+21 1.571e+21
Step  200/ 200  dt=1.292e-15  Mass=9.30251829e+04  dM=4.195e-01  Energy=2.28094355e+21  dE=9.944e+15  spectrum[0..7]: inf inf inf inf inf inf inf inf
Center (rho,mx,my,Bx,By,E): 0.996392 0.015229 -0.008865 0.009325 0.028031 2.479517

# convert dumped frames (frames/*.bin) to .npy and animate
python - <<'PY'
import numpy as np, glob
NX=256; NY=256
for f in glob.glob("frames/*.bin"):
    arr = np.fromfile(f, dtype=np.float32).reshape(NY, NX)
    np.save(f.replace(".bin", ".npy"), arr)
PY

python animate_poisson.py --pattern "frames/rho_*.npy" --mode heatmap --title "rho"
python animate_poisson.py --pattern "frames/phi_*.npy" --mode surface --title "phi"

```
After using `-c` flag, we get an object file and instead may need to relink using `nvcc mhd_poisson_tiled -lcufft -o mhd_poisson_tiled.exe` or use `nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v mhd2d_adv.cu -lcufft -o mhd_poisson_tiled` to relink.



Day 13:
Write a 32×32 butterfly FFT kernel in PTX; compare its runtime with cuFFT 

013_fft_ptx_butterfly

Goals:
- Code a 32x32 butterfly FFT kernel (hand-written PTX or inline PTX) and compare against cuFFT.
- Cross-check with ideas from NVIDIA's "FFT on GPUs" blog (but implement locally).
- Collect PTX/SASS and basic timing to see instruction mix and occupancy.

Suggested workflow:
1) Implement the custom kernel in `butterfly_fft.cu` (32-point or 32x32 tile; keep IO simple, e.g., in-place complex float).
2) Build and dump PTX/SASS:
   - `nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v butterfly_fft.cu -lcufft -o butterfly_fft`
   - `nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -ptx butterfly_fft.cu -o butterfly_fft.ptx`
   - `cuobjdump --dump-sass butterfly_fft > butterfly_fft.sass`
3) Compare against cuFFT baseline:
   - Build and run: `nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v butterfly_fft.cu -lcufft -o butterfly_fft`
   - `./butterfly_fft 1` prints PTX-FFT vs cuFFT timings and max |err|; increase tiles for steadier timing.
4) Capture notes on instruction patterns and occupancy:
   - Record ptxas info (regs, smem, barriers) from `-Xptxas -v`.
   - Skim SASS for shuffle/twiddle patterns and the shared-memory transpose.
5) Animate the butterfly stages (similar to Day 12):
   - `./butterfly_fft --dump` (writes `frames/fft_stage_00.bin` ... `frames/fft_stage_13.bin`)
   - `python animate_fft.py --pattern "frames/fft_stage_*.bin" --field mag --mode heatmap --out fft_mag.mp4 --title "FFT magnitude"`
   - `python animate_fft.py --pattern "frames/fft_stage_*.bin" --field phase --mode heatmap --out fft_phase.mp4 --title "FFT phase"`
   - Stage map: 00 input, 01-05 row stages, 06 row output (bit-reversed), 07 column input (transpose), 08-12 column stages, 13 final output.


```bash 
>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v butterfly_fft.cu -lcufft -o butterfly_fft
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z17fft32x32_warp_ptxPK6float2PS_i' for 'sm_89'
ptxas info    : Function properties for _Z17fft32x32_warp_ptxPK6float2PS_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 27 registers, used 1 barriers, 8448 bytes smem, 372 bytes cmem[0]
ptxas info    : Compile time = 5.922 ms

>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -ptx butterfly_fft.cu -o butterfly_fft.ptx
>> cuobjdump --dump-sass butterfly_fft > butterfly_fft.sass

>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v butterfly_fft.cu -lcufft -o butterfly_fft
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z23fft32x32_warp_ptx_debugPK6float2PS_S2_i' for 'sm_89'
ptxas info    : Function properties for _Z23fft32x32_warp_ptx_debugPK6float2PS_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 38 registers, used 1 barriers, 8448 bytes smem, 380 bytes cmem[0]
ptxas info    : Compile time = 33.918 ms
ptxas info    : Compiling entry function '_Z17fft32x32_warp_ptxPK6float2PS_i' for 'sm_89'
ptxas info    : Function properties for _Z17fft32x32_warp_ptxPK6float2PS_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 27 registers, used 1 barriers, 8448 bytes smem, 372 bytes cmem[0]
ptxas info    : Compile time = 5.909 ms
```

Day 14:
Switch MHD solver to FP16 + Tensor Cores; verify HMMA instructions in PTX

What this demo does:
- Uses a 16x16 WMMA tile blur (FP16) to update rho (density) on each step.
- Solves Poisson (cuFFT) to compute phi from rho each step.
- Dumps rho/phi every step to `frames/` so you can animate both in one frame.
- Periodic domain, sinusoidal + Gaussian rho initialization (MHD-style seeding).
- Animate rho + phi side-by-side to compare fluid density vs potential.
- Compute phi each step with cuFFT Poisson (from Day 12) and dump rho/phi frames.


Suggested workflow:
Build:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 mhd_fp16_tc.cu -lcufft -o mhd_fp16_tc
```

Run (NX NY STEPS mix phi_scale):
```bash
./mhd_fp16_tc 128 128 60
```

Verify Tensor Core instructions:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 -ptx mhd_fp16_tc.cu -o mhd_fp16_tc.ptx
rg -n "mma.sync" mhd_fp16_tc.ptx
cuobjdump --dump-sass mhd_fp16_tc | rg "HMMA|mma"
```

Animate rho + phi in a single frame:
```bash
python animate_mhd_fp16.py --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" \
  --shape 128 128 --out mhd_rho_phi.mp4 --fps 12 --title "FP16 MHD (rho | phi)"
```
If `ffmpeg` is missing, use `--out mhd_rho_phi.gif`.

Animate rho + phi as 3D surfaces (time t shown in title):
```bash
python animate_mhd_fp16.py --mode surface --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" \
  --shape 128 128 --out mhd_rho_phi_3d.mp4 --fps 12 --dt 1.0 --stride 2 \
  --title "FP16 MHD 3D (rho | phi)"
```

Advanced (014_5_advanced_mhd_fp16_tensorcore):
- Full Rusanov MHD update in FP16 storage + Poisson coupling + WMMA smoothing.
- Combined 3D visualization: rho surface colored by phi.
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 mhd_fp16_tc_adv.cu -lcufft -o mhd_fp16_tc_adv
./mhd_fp16_tc_adv 192 192 120 1.0 0.08 0.05 1
python animate_mhd_advanced.py --mode combined --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" \
  --shape 192 192 --out mhd_combined_3d.mp4 --fps 12 --dt 1.0 --stride 2
```


```bash
>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 mhd_fp16_tc.cu -lcufft -o mhd_fp16_tc
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z15update_rho_wmmaPK6__halfPS_S1_PKfiiff' for 'sm_89'
ptxas info    : Function properties for _Z15update_rho_wmmaPK6__halfPS_S1_PKfiiff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 28 registers, used 1 barriers, 1024 bytes smem, 400 bytes cmem[0]
ptxas info    : Compile time = 3.144 ms
ptxas info    : Compiling entry function '_Z10reduce_sumPKfPdi' for 'sm_89'
ptxas info    : Function properties for _Z10reduce_sumPKfPdi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 1 barriers, 2048 bytes smem, 372 bytes cmem[0]
ptxas info    : Compile time = 1.221 ms
ptxas info    : Compiling entry function '_Z13float_to_halfPKfP6__halfi' for 'sm_89'
ptxas info    : Function properties for _Z13float_to_halfPKfP6__halfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.597 ms
ptxas info    : Compiling entry function '_Z13half_to_floatPK6__halfPfi' for 'sm_89'
ptxas info    : Function properties for _Z13half_to_floatPK6__halfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.544 ms
ptxas info    : Compiling entry function '_Z19spectrum_power_binsPK6float2iiPfPji' for 'sm_89'
ptxas info    : Function properties for _Z19spectrum_power_binsPK6float2iiPfPji
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 1.838 ms
ptxas info    : Compiling entry function '_Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i' for 'sm_89'
ptxas info    : Function properties for _Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 17 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 3.292 ms
ptxas info    : Compiling entry function '_Z15complex_to_realPfPK6float2if' for 'sm_89'
ptxas info    : Function properties for _Z15complex_to_realPfPK6float2if
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 0.496 ms
ptxas info    : Compiling entry function '_Z15real_to_complexP6float2PKfi' for 'sm_89'
ptxas info    : Function properties for _Z15real_to_complexP6float2PKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.448 ms
ptxas info    : Compiling entry function '_Z14apply_grad_phiPfS_PKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14apply_grad_phiPfS_PKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, used 0 barriers, 396 bytes cmem[0]
ptxas info    : Compile time = 4.741 ms
ptxas info    : Compiling entry function '_Z13subtract_meanPfif' for 'sm_89'
ptxas info    : Function properties for _Z13subtract_meanPfif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 0.553 ms
ptxas info    : Compiling entry function '_Z21apply_poisson_scalingP6float2PKfii' for 'sm_89'
ptxas info    : Function properties for _Z21apply_poisson_scalingP6float2PKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 3.334 ms
ptxas info    : Compiling entry function '_Z7fill_k2Pfii' for 'sm_89'
ptxas info    : Function properties for _Z7fill_k2Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 368 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compile time = 0.737 ms

>> ./mhd_fp16_tc 128 128 60
Step    0/  60  rho[min,max]=[0.9199, 1.6562]
Step   10/  60  rho[min,max]=[0.9199, 1.6494]
Step   20/  60  rho[min,max]=[0.9199, 1.6455]
Step   30/  60  rho[min,max]=[0.9199, 1.6455]
Step   40/  60  rho[min,max]=[0.9199, 1.6455]
Step   50/  60  rho[min,max]=[0.9199, 1.6455]
Step   59/  60  rho[min,max]=[0.9199, 1.6455]

>> python animate_mhd_fp16.py --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" --shape 128 128 --out mhd_rho_phi.mp4 --fps 12
Saved animation to mhd_rho_phi.mp4 using 60 frames.

>> grep -n "mma.sync" mhd_fp16_tc.ptx
831:    wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {%f4, %f5, %f6, %f7, %f8, %f9, %f10, %f11}, {%r14, %r15, %r16, %r17, %r18, %r19, %r20, %r21}, {%r23, %r24, %r25, %r26, %r27, %r28, %r29, %r30}, {%f3, %f3, %f3, %f3, %f3, %f3, %f3, %f3};

>> cuobjdump --dump-sass mhd_fp16_tc | grep -E "HMMA|MMA"
        /*02f0*/                   HMMA.16816.F32 R12, R4.reuse, R22, RZ ;              /* 0x00000016040c723c */
        /*0300*/                   HMMA.16816.F32 R4, R4, R24, RZ ;                     /* 0x000000180404723c */
```

Day 14.5:
Goal:
- Combining all earlier days
- Full MHD Rusanov update using FP16 storage, Poisson coupling for phi, and WMMA Tensor Core smoothing.

What this does:
- Evolves 2D MHD state (rho, mx, my, Bx, By, E) with a first-order Rusanov flux.
- Stores state in FP16 (half) and computes fluxes in FP32.
- Solves Poisson (cuFFT) for phi from rho each step, applies grad(phi) to momentum.
- Runs a WMMA tile blur on rho to exercise Tensor Cores (HMMA/mma.sync).
- Dumps rho/phi every step to `frames/` for animation.

Build:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 mhd_fp16_tc_adv.cu -lcufft -o mhd_fp16_tc_adv
```

Run (NX NY STEPS dt_scale phi_scale wmma_mix dump_every):
```bash
./mhd_fp16_tc_adv 192 192 120 1.0 0.08 0.05 1
```

Verify Tensor Core instructions:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 -ptx mhd_fp16_tc_adv.cu -o mhd_fp16_tc_adv.ptx
rg -n "mma.sync" mhd_fp16_tc_adv.ptx
cuobjdump --dump-sass mhd_fp16_tc_adv | rg "HMMA|mma"
```
If `rg` is missing, use `grep -n "mma.sync" mhd_fp16_tc_adv.ptx` and `cuobjdump --dump-sass mhd_fp16_tc_adv | grep -E "HMMA|MMA"`.

Animate combined 3D (rho as surface, phi as color):
```bash
python animate_mhd_advanced.py --mode combined \
  --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" --shape 192 192 \
  --out mhd_combined_3d.mp4 --fps 12 --dt 1.0 --stride 2 \
  --title "Advanced MHD (rho colored by phi)"
```

Side-by-side heatmaps:
```bash
python animate_mhd_advanced.py --mode side-by-side \
  --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" --shape 192 192 \
  --out mhd_side_by_side.mp4 --fps 12 --dt 1.0
```

Notes:
- NX and NY must be multiples of 16 for WMMA tiles.
- Each saved frame corresponds to one step (step size = 1 by default).
- Increase `phi_scale` or `wmma_mix` for more visible dynamics.


```bash
>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 mhd_fp16_tc_adv.cu -lcufft -o mhd_fp16_tc_adv
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z13wmma_blur_rhoP6__halfPKS_iif' for 'sm_89'
ptxas info    : Function properties for _Z13wmma_blur_rhoP6__halfPKS_iif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 28 registers, used 1 barriers, 1024 bytes smem, 380 bytes cmem[0]
ptxas info    : Compile time = 22.399 ms
ptxas info    : Compiling entry function '_Z10reduce_sumPKfPdi' for 'sm_89'
ptxas info    : Function properties for _Z10reduce_sumPKfPdi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 1 barriers, 2048 bytes smem, 372 bytes cmem[0]
ptxas info    : Compile time = 1.414 ms
ptxas info    : Compiling entry function '_Z19apply_grad_phi_halfP6__halfPKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z19apply_grad_phi_halfP6__halfPKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 8.226 ms
ptxas info    : Compiling entry function '_Z25extract_rho_half_to_floatPK6__halfPfi' for 'sm_89'
ptxas info    : Function properties for _Z25extract_rho_half_to_floatPK6__halfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.786 ms
ptxas info    : Compiling entry function '_Z20kernel_maxspeed_halfPK6__halfii' for 'sm_89'
ptxas info    : Function properties for _Z20kernel_maxspeed_halfPK6__halfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 21 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 6.022 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.305 ms
ptxas info    : Compiling entry function '_Z21step_mhd_rusanov_halfPK6__halfPS_iifff' for 'sm_89'
ptxas info    : Function properties for _Z21step_mhd_rusanov_halfPK6__halfPS_iifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 72 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 21.239 ms
ptxas info    : Compiling entry function '_Z19spectrum_power_binsPK6float2iiPfPji' for 'sm_89'
ptxas info    : Function properties for _Z19spectrum_power_binsPK6float2iiPfPji
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 2.069 ms
ptxas info    : Compiling entry function '_Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i' for 'sm_89'
ptxas info    : Function properties for _Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 17 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 3.481 ms
ptxas info    : Compiling entry function '_Z15complex_to_realPfPK6float2if' for 'sm_89'
ptxas info    : Function properties for _Z15complex_to_realPfPK6float2if
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 0.554 ms
ptxas info    : Compiling entry function '_Z15real_to_complexP6float2PKfi' for 'sm_89'
ptxas info    : Function properties for _Z15real_to_complexP6float2PKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.675 ms
ptxas info    : Compiling entry function '_Z14apply_grad_phiPfS_PKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14apply_grad_phiPfS_PKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, used 0 barriers, 396 bytes cmem[0]
ptxas info    : Compile time = 5.492 ms
ptxas info    : Compiling entry function '_Z13subtract_meanPfif' for 'sm_89'
ptxas info    : Function properties for _Z13subtract_meanPfif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 0.621 ms
ptxas info    : Compiling entry function '_Z21apply_poisson_scalingP6float2PKfii' for 'sm_89'
ptxas info    : Function properties for _Z21apply_poisson_scalingP6float2PKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 3.435 ms
ptxas info    : Compiling entry function '_Z7fill_k2Pfii' for 'sm_89'
ptxas info    : Function properties for _Z7fill_k2Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 368 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compile time = 0.836 ms

>> ./mhd_fp16_tc_adv 192 192 120 1.0 0.08 0.05 1
Step    0/ 120  dt=4.579e-04  rho[min,max]=[1.0000, 1.0000]
Step   10/ 120  dt=4.577e-04  rho[min,max]=[0.9976, 1.0049]
Step   20/ 120  dt=4.570e-04  rho[min,max]=[0.9912, 1.0195]
Step   30/ 120  dt=4.561e-04  rho[min,max]=[0.9810, 1.0420]
Step   40/ 120  dt=4.545e-04  rho[min,max]=[0.9673, 1.0752]
Step   50/ 120  dt=4.455e-04  rho[min,max]=[0.9512, 1.1172]
Step   60/ 120  dt=4.376e-04  rho[min,max]=[0.9316, 1.1689]
Step   70/ 120  dt=4.301e-04  rho[min,max]=[0.9121, 1.2305]
Step   80/ 120  dt=4.231e-04  rho[min,max]=[0.8887, 1.3008]
Step   90/ 120  dt=4.163e-04  rho[min,max]=[0.8643, 1.3799]
Step  100/ 120  dt=4.097e-04  rho[min,max]=[0.8398, 1.4678]
Step  110/ 120  dt=4.030e-04  rho[min,max]=[0.8145, 1.5645]
Step  119/ 120  dt=3.966e-04  rho[min,max]=[0.7900, 1.6553]

>> python animate_mhd_advanced.py --mode combined \
  --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" --shape 192 192 \
  --out mhd_combined_3d.mp4 --fps 12 --dt 1.0 --stride 2


>> python animate_mhd_advanced.py --mode side-by-side \
  --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" --shape 192 192 \
  --out mhd_side_by_side.mp4 --fps 12 --dt 1.0

>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 -ptx mhd_fp16_tc_adv.cu -o mhd_fp16_tc_adv.ptx

>> grep -n "mma.sync" mhd_fp16_tc_adv.ptx
2021:   wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {%f3, %f4, %f5, %f6, %f7, %f8, %f9, %f10}, {%r14, %r15, %r16, %r17, %r18, %r19, %r20, %r21}, {%r23, %r24, %r25, %r26, %r27, %r28, %r29, %r30}, {%f2, %f2, %f2, %f2, %f2, %f2, %f2, %f2};

>> cuobjdump --dump-sass mhd_fp16_tc_adv | grep -E "HMMA|MMA"
        /*02f0*/                   HMMA.16816.F32 R12, R4.reuse, R22, RZ ;               /* 0x00000016040c723c */
        /*0300*/                   HMMA.16816.F32 R4, R4, R24, RZ ;                      /* 0x000000180404723c */
```

Day 15:
Inline approximate reciprocal (`rcp.approx.f32`) PTX in the flux loop to replace division 

- Inline `rcp.approx.f32` in the flux loop to replace division by rho (PTX ISA v8).
- Compare exact vs approximate reciprocal runs with a combined 3D animation.
- Animate exact vs approx to visualize drift.


What this demo does:
- Runs full MHD Rusanov updates with FP16 storage (like Day 14.5).
- Uses inline PTX `rcp.approx.f32` for the approximate run in flux/pressure/speed paths.
- Solves Poisson for phi each step and dumps rho/phi frames for exact + approx.
- Outputs two frame folders: `frames_exact/` and `frames_approx/`.

Build:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 mhd_fp16_tc_rcp.cu -lcufft -o mhd_fp16_tc_rcp
```

Run (NX NY STEPS dt_scale phi_scale wmma_mix dump_every):
```bash
./mhd_fp16_tc_rcp 192 192 120 1.0 0.08 0.05 1
```

Verify `rcp.approx.f32` in PTX:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 -ptx mhd_fp16_tc_rcp.cu -o mhd_fp16_tc_rcp.ptx
grep -n "rcp.approx.f32" mhd_fp16_tc_rcp.ptx
```

Compare exact vs rcp.approx (3D combined view):
```bash
python animate_mhd_rcp_compare.py --mode compare \
  --exact-rho "frames_exact/rho_*.bin" --exact-phi "frames_exact/phi_*.bin" \
  --approx-rho "frames_approx/rho_*.bin" --approx-phi "frames_approx/phi_*.bin" \
  --shape 192 192 --out mhd_rcp_compare.mp4 --fps 12 --dt 1.0 --stride 2
```

Optional: visualize rho difference heatmap:
```bash
python animate_mhd_rcp_compare.py --mode delta --shape 192 192 --out mhd_rcp_delta.mp4
```

Execution:

```bash
>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 mhd_fp16_tc_rcp.cu -lcufft -o mhd_fp16_tc_rcp
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z21step_mhd_rusanov_halfILb1EEvPK6__halfPS0_iifff' for 'sm_89'
ptxas info    : Function properties for _Z21step_mhd_rusanov_halfILb1EEvPK6__halfPS0_iifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 63 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 47.974 ms
ptxas info    : Compiling entry function '_Z20kernel_maxspeed_halfILb1EEvPK6__halfii' for 'sm_89'
ptxas info    : Function properties for _Z20kernel_maxspeed_halfILb1EEvPK6__halfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 17 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 2.271 ms
ptxas info    : Compiling entry function '_Z21step_mhd_rusanov_halfILb0EEvPK6__halfPS0_iifff' for 'sm_89'
ptxas info    : Function properties for _Z21step_mhd_rusanov_halfILb0EEvPK6__halfPS0_iifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 65 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 16.345 ms
ptxas info    : Compiling entry function '_Z20kernel_maxspeed_halfILb0EEvPK6__halfii' for 'sm_89'
ptxas info    : Function properties for _Z20kernel_maxspeed_halfILb0EEvPK6__halfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 18 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 3.292 ms
ptxas info    : Compiling entry function '_Z13wmma_blur_rhoP6__halfPKS_iif' for 'sm_89'
ptxas info    : Function properties for _Z13wmma_blur_rhoP6__halfPKS_iif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 28 registers, used 1 barriers, 1024 bytes smem, 380 bytes cmem[0]
ptxas info    : Compile time = 1.795 ms
ptxas info    : Compiling entry function '_Z10reduce_sumPKfPdi' for 'sm_89'
ptxas info    : Function properties for _Z10reduce_sumPKfPdi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 1 barriers, 2048 bytes smem, 372 bytes cmem[0]
ptxas info    : Compile time = 1.071 ms
ptxas info    : Compiling entry function '_Z19apply_grad_phi_halfP6__halfPKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z19apply_grad_phi_halfP6__halfPKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 4.894 ms
ptxas info    : Compiling entry function '_Z25extract_rho_half_to_floatPK6__halfPfi' for 'sm_89'
ptxas info    : Function properties for _Z25extract_rho_half_to_floatPK6__halfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.563 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.293 ms
ptxas info    : Compiling entry function '_Z19spectrum_power_binsPK6float2iiPfPji' for 'sm_89'
ptxas info    : Function properties for _Z19spectrum_power_binsPK6float2iiPfPji
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 1.960 ms
ptxas info    : Compiling entry function '_Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i' for 'sm_89'
ptxas info    : Function properties for _Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 17 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 2.933 ms
ptxas info    : Compiling entry function '_Z15complex_to_realPfPK6float2if' for 'sm_89'
ptxas info    : Function properties for _Z15complex_to_realPfPK6float2if
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 0.507 ms
ptxas info    : Compiling entry function '_Z15real_to_complexP6float2PKfi' for 'sm_89'
ptxas info    : Function properties for _Z15real_to_complexP6float2PKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.852 ms
ptxas info    : Compiling entry function '_Z14apply_grad_phiPfS_PKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14apply_grad_phiPfS_PKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, used 0 barriers, 396 bytes cmem[0]
ptxas info    : Compile time = 4.888 ms
ptxas info    : Compiling entry function '_Z13subtract_meanPfif' for 'sm_89'
ptxas info    : Function properties for _Z13subtract_meanPfif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 0.484 ms
ptxas info    : Compiling entry function '_Z21apply_poisson_scalingP6float2PKfii' for 'sm_89'
ptxas info    : Function properties for _Z21apply_poisson_scalingP6float2PKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 3.106 ms
ptxas info    : Compiling entry function '_Z7fill_k2Pfii' for 'sm_89'
ptxas info    : Function properties for _Z7fill_k2Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 368 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compile time = 0.755 ms

>> ./mhd_fp16_tc_rcp 192 192 120 1.0 0.08 0.05 1

>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 -ptx mhd_fp16_tc_rcp.cu -o mhd_fp16_tc_rcp.ptx

>> grep -n "rcp.approx.f32" mhd_fp16_tc_rcp.ptx
2194:   rcp.approx.f32 %f7, %f12;
2204:   rcp.approx.f32 %f9, %f12;
2230:   rcp.approx.f32 %f11, %f12;
2660:   rcp.approx.f32 %f34, %f51;
2670:   rcp.approx.f32 %f36, %f51;
2720:   rcp.approx.f32 %f38, %f111;
2730:   rcp.approx.f32 %f40, %f111;
2777:   rcp.approx.f32 %f42, %f51;
2783:   rcp.approx.f32 %f44, %f111;
2790:   rcp.approx.f32 %f46, %f51;
2811:   rcp.approx.f32 %f48, %f111;
2832:   rcp.approx.f32 %f50, %f51;
2847:   rcp.approx.f32 %f52, %f111;
2903:   rcp.approx.f32 %f54, %f111;
2913:   rcp.approx.f32 %f56, %f111;
2957:   rcp.approx.f32 %f58, %f73;
2967:   rcp.approx.f32 %f60, %f73;
3013:   rcp.approx.f32 %f62, %f111;
3019:   rcp.approx.f32 %f64, %f73;
3026:   rcp.approx.f32 %f66, %f111;
3047:   rcp.approx.f32 %f68, %f73;
3068:   rcp.approx.f32 %f70, %f111;
3082:   rcp.approx.f32 %f72, %f73;
3140:   rcp.approx.f32 %f74, %f91;
3150:   rcp.approx.f32 %f76, %f91;
3197:   rcp.approx.f32 %f78, %f111;
3207:   rcp.approx.f32 %f80, %f111;
3247:   rcp.approx.f32 %f82, %f91;
3253:   rcp.approx.f32 %f84, %f111;
3260:   rcp.approx.f32 %f86, %f91;
3281:   rcp.approx.f32 %f88, %f111;
3302:   rcp.approx.f32 %f90, %f91;
3316:   rcp.approx.f32 %f92, %f111;
3371:   rcp.approx.f32 %f94, %f111;
3381:   rcp.approx.f32 %f96, %f111;
3425:   rcp.approx.f32 %f98, %f113;
3435:   rcp.approx.f32 %f100, %f113;
3481:   rcp.approx.f32 %f102, %f111;
3487:   rcp.approx.f32 %f104, %f113;
3494:   rcp.approx.f32 %f106, %f111;
3515:   rcp.approx.f32 %f108, %f113;
3536:   rcp.approx.f32 %f110, %f111;
3550:   rcp.approx.f32 %f112, %f113;

>> python animate_mhd_rcp_compare.py --mode compare \
  --exact-rho "frames_exact/rho_*.bin" --exact-phi "frames_exact/phi_*.bin" \
  --approx-rho "frames_approx/rho_*.bin" --approx-phi "frames_approx/phi_*.bin" \
  --shape 192 192 --out mhd_rcp_compare.mp4 --fps 12 --dt 1.0 --stride 2
Saved animation to mhd_rcp_compare.mp4 using 120 frames.

>> python animate_mhd_rcp_compare.py --mode delta --shape 192 192 --out mhd_rcp_delta.mp4
Saved animation to mhd_rcp_delta.mp4 using 120 frames.
```

**Notes:**
- NX and NY must be multiples of 16 for WMMA tiles.
- Each saved frame corresponds to one step (step size = 1 by default).
- The approximate reciprocal affects flux/pressure/speed calculations, so dynamics may drift.



-- Created a Poetry environment for my HPC experiments
Dive into HPC libraries: install and test cuBLAS and cuRAND with small matrix multiply and random number generation examples.
Reference: cuBLAS Documentation, cuRAND Documentation
```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

int main() {
    // 1) cuBLAS Example: Simple matrix multiply
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int N = 2;
    float hA[N*N] = {1, 2, 3, 4};
    float hB[N*N] = {5, 6, 7, 8};
    float hC[N*N] = {0, 0, 0, 0};

    float *dA, *dB, *dC;
    cudaMalloc(&dA, N*N*sizeof(float));
    cudaMalloc(&dB, N*N*sizeof(float));
    cudaMalloc(&dC, N*N*sizeof(float));

    cudaMemcpy(dA, hA, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N*N*sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;

    // C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                dA, N,
                dB, N,
                &beta,
                dC, N);

    cudaMemcpy(hC, dC, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Result C[0] = " << hC[0] << std::endl; // Expect 19 (1*5 + 2*7)

    cublasDestroy(handle);

    // 2) cuRAND Example: generate random numbers on GPU
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 123ULL);

    float* dRand;
    cudaMalloc(&dRand, N*N*sizeof(float));

    // Generate uniform random numbers in [0, 1)
    curandGenerateUniform(gen, dRand, N*N);

    float hRand[N*N];
    cudaMemcpy(hRand, dRand, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Random number sample: " << hRand[0] << std::endl;

    curandDestroyGenerator(gen);
    cudaFree(dRand);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
```
```bash
nvcc -lcublas -lcurand -o cublas_curand_example cublas_curand_example.cu
./cublas_curand_example
```

Day 16:
Put the full time-step in a CUDA Graph; run 1000 iterations in one graph launch
- Capture the full time-step in a CUDA Graph and run 1000 iterations in one graph launch.
- Dump rho/phi frames and animate with the combined 3D view as Day 15.

What this does:
- FP16 storage MHD Rusanov update with Poisson coupling (cuFFT) and optional WMMA smoothing.
- Uses CUDA Graph capture to unroll `STEPS` iterations into one graph.
- Stores frames on-device during the graph, then copies out after launch.

Build:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 mhd_fp16_tc_graph.cu -lcufft -o mhd_fp16_tc_graph
```

Run (NX NY STEPS dt_scale phi_scale wmma_mix dump_every):
```bash
./mhd_fp16_tc_graph 128 128 1000 1.0 0.08 0.05 1
```

Animate combined 3D (rho surface colored by phi):
```bash
python animate_mhd_advanced.py --mode combined \
  --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" --shape 128 128 \
  --out mhd_graph_combined_3d.mp4 --fps 12 --dt 1.0 --stride 2 \
  --title "CUDA Graph MHD (rho colored by phi)"
```

Notes:
- NX and NY must be multiples of 16 for WMMA tiles.
- `dump_every=1` stores every step; reduce if memory is tight.
- Frames are written after the graph completes (device -> host).

Execution:
```bash
>> nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 mhd_fp16_tc_graph.cu -lcufft -o mhd_fp16_tc_graph
ptxas info    : 4 bytes gmem, 8 bytes cmem[4]
ptxas info    : Compiling entry function '_Z15complex_to_realPfPK6float2if' for 'sm_89'
ptxas info    : Function properties for _Z15complex_to_realPfPK6float2if
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 1.767 ms
ptxas info    : Compiling entry function '_Z15real_to_complexP6float2PKfi' for 'sm_89'
ptxas info    : Function properties for _Z15real_to_complexP6float2PKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.608 ms
ptxas info    : Compiling entry function '_Z20subtract_mean_devicePfiPKf' for 'sm_89'
ptxas info    : Function properties for _Z20subtract_mean_devicePfiPKf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 0.509 ms
ptxas info    : Compiling entry function '_Z21apply_poisson_scalingP6float2PKfii' for 'sm_89'
ptxas info    : Function properties for _Z21apply_poisson_scalingP6float2PKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 3.473 ms
ptxas info    : Compiling entry function '_Z7fill_k2Pfii' for 'sm_89'
ptxas info    : Function properties for _Z7fill_k2Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 0.830 ms
ptxas info    : Compiling entry function '_Z11store_framePKfS0_PfS1_ii' for 'sm_89'
ptxas info    : Function properties for _Z11store_framePKfS0_PfS1_ii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compile time = 0.607 ms
ptxas info    : Compiling entry function '_Z13wmma_blur_rhoP6__halfPKS_iif' for 'sm_89'
ptxas info    : Function properties for _Z13wmma_blur_rhoP6__halfPKS_iif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 28 registers, used 1 barriers, 1024 bytes smem, 380 bytes cmem[0]
ptxas info    : Compile time = 1.846 ms
ptxas info    : Compiling entry function '_Z19compute_mean_kernelPKdPfi' for 'sm_89'
ptxas info    : Function properties for _Z19compute_mean_kernelPKdPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 23 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 2.798 ms
ptxas info    : Compiling entry function '_Z10reduce_sumPKfPdi' for 'sm_89'
ptxas info    : Function properties for _Z10reduce_sumPKfPdi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 1 barriers, 2048 bytes smem, 372 bytes cmem[0]
ptxas info    : Compile time = 1.721 ms
ptxas info    : Compiling entry function '_Z25apply_grad_phi_half_graphP6__halfPKfS2_iifff' for 'sm_89'
ptxas info    : Function properties for _Z25apply_grad_phi_half_graphP6__halfPKfS2_iifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, used 0 barriers, 396 bytes cmem[0]
ptxas info    : Compile time = 5.665 ms
ptxas info    : Compiling entry function '_Z25extract_rho_half_to_floatPK6__halfPfi' for 'sm_89'
ptxas info    : Function properties for _Z25extract_rho_half_to_floatPK6__halfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.523 ms
ptxas info    : Compiling entry function '_Z17compute_dt_kernelPffff' for 'sm_89'
ptxas info    : Function properties for _Z17compute_dt_kernelPffff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 2.816 ms
ptxas info    : Compiling entry function '_Z20kernel_maxspeed_halfPK6__halfii' for 'sm_89'
ptxas info    : Function properties for _Z20kernel_maxspeed_halfPK6__halfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 21 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 5.455 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.384 ms
ptxas info    : Compiling entry function '_Z27step_mhd_rusanov_half_graphPK6__halfPS_PKfiiff' for 'sm_89'
ptxas info    : Function properties for _Z27step_mhd_rusanov_half_graphPK6__halfPS_PKfiiff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 72 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compile time = 21.057 ms

>> ./mhd_fp16_tc_graph 128 128 1000 1.0 0.08 0.05 1
>> python animate_mhd_advanced.py --mode combined \
  --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" --shape 128 128 \
  --out mhd_graph_combined_3d.mp4 --fps 12 --dt 1.0 --stride 2 \
  --title "CUDA Graph MHD (rho colored by phi)"


```

Day 17: Add dynamic parallelism: launch a sub-grid refinement kernel from within the solver
- Launch a refine kernel from inside the solver when rho gradients exceed a threshold.
- Dump rho/phi/bx/by frames for base vs refined and compare with magnetic field annotations.

Build (run from 017_dynpar_mhd):
```bash
nvcc -O3 -arch=sm_89 -rdc=true -lineinfo -Xptxas -v 017_dynpar_mhd.cu -lcudadevrt -lcufft -o mhd_dynpar
```

Run (NX NY STEPS refine_thresh dump_every):
```bash
./mhd_dynpar 128 128 120 0.15 1
```

Animate comparison:
```bash
python animate_dynpar_compare.py --base frames_base --refine frames_refine \
  --shape 128 128 --out dynpar_compare.mp4 --fps 12 --stride 6
```

Earth-style 3D compare (dynpar vs 014_5):
- Requires baseline frames in `../014_5_advanced_mhd_fp16_tensorcore/frames`.
- Overlays rotating dipole field lines around an Earth sphere while rho/phi share the same plane.
- If frame counts differ, the script uses the minimum length.

```bash
python animate_dynpar_earth_compare.py \
  --dynpar-rho "frames_refine/rho_*.bin" --dynpar-phi "frames_refine/phi_*.bin" \
  --dynpar-bx "frames_refine/bx_*.bin" --dynpar-by "frames_refine/by_*.bin" \
  --baseline-rho "../014_5_advanced_mhd_fp16_tensorcore/frames/rho_*.bin" \
  --baseline-phi "../014_5_advanced_mhd_fp16_tensorcore/frames/phi_*.bin" \
  --dynpar-shape 128 128 --baseline-shape 192 192 \
  --out dynpar_earth_compare_3d.mp4 --fps 12 --dt 1.0 --stride 2 \
  --tilt-deg 20 --spin-deg 2.5 --line-shells 5 --line-phis 8 --line-scale 0.42
```

Execution:
```bash
>> nvcc -O3 -arch=sm_89 -rdc=true -lineinfo -Xptxas -v 017_dynpar_mhd.cu -lcudadevrt -lcufft -o mhd_dynpar
nvcc -O3 -arch=sm_89 -rdc=true -lineinfo -Xptxas -v 017_dynpar_mhd.cu -lcudadevrt -lcufft -o mhd_dynpar
ptxas info    : 4 bytes gmem
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 30 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 3.934 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.445 ms
ptxas info    : Compiling entry function '_Z23step_mhd_rusanov_dynparPKfPfiiffff' for 'sm_89'
ptxas info    : Function properties for _Z23step_mhd_rusanov_dynparPKfPfiiffff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 91 registers, used 1 barriers, 392 bytes cmem[0]
ptxas info    : Compile time = 21.015 ms
ptxas info    : Compiling entry function '_Z12refine_patchPKfPfiifffiiii' for 'sm_89'
ptxas info    : Function properties for _Z12refine_patchPKfPfiifffiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 0 barriers, 404 bytes cmem[0]
ptxas info    : Compile time = 16.140 ms
ptxas info    : Compiling entry function '_Z19spectrum_power_binsPK6float2iiPfPji' for 'sm_89'
ptxas info    : Function properties for _Z19spectrum_power_binsPK6float2iiPfPji
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 1.683 ms
ptxas info    : Compiling entry function '_Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i' for 'sm_89'
ptxas info    : Function properties for _Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 1.653 ms
ptxas info    : Compiling entry function '_Z15complex_to_realPfPK6float2if' for 'sm_89'
ptxas info    : Function properties for _Z15complex_to_realPfPK6float2if
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 0.683 ms
ptxas info    : Compiling entry function '_Z15real_to_complexP6float2PKfi' for 'sm_89'
ptxas info    : Function properties for _Z15real_to_complexP6float2PKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.592 ms
ptxas info    : Compiling entry function '_Z14apply_grad_phiPfS_PKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14apply_grad_phiPfS_PKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 396 bytes cmem[0]
ptxas info    : Compile time = 3.157 ms
ptxas info    : Compiling entry function '_Z13subtract_meanPfif' for 'sm_89'
ptxas info    : Function properties for _Z13subtract_meanPfif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 0.468 ms
ptxas info    : Compiling entry function '_Z21apply_poisson_scalingP6float2PKfii' for 'sm_89'
ptxas info    : Function properties for _Z21apply_poisson_scalingP6float2PKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 1.135 ms
ptxas info    : Compiling entry function '_Z7fill_k2Pfii' for 'sm_89'
ptxas info    : Function properties for _Z7fill_k2Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 368 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compile time = 0.740 ms

>> ./mhd_dynpar 128 128 120 0.15 1
>> python animate_dynpar_compare.py --base frames_base --refine frames_refine \
  --shape 128 128 --out dynpar_compare.mp4 --fps 12 --stride 6
/mnt/c/Users/anshm/250DaysStraight/017_dynpar_mhd/animate_dynpar_compare.py:77: MatplotlibDeprecationWarning: The collections attribute was deprecated in Matplotlib 3.8 and will be removed in 3.10.
  for col in c1.collections:
/mnt/c/Users/anshm/250DaysStraight/017_dynpar_mhd/animate_dynpar_compare.py:79: MatplotlibDeprecationWarning: The collections attribute was deprecated in Matplotlib 3.8 and will be removed in 3.10.
  for col in c2.collections:

>> python animate_dynpar_earth_compare.py \
  --dynpar-rho "frames_refine/rho_*.bin" --dynpar-phi "frames_refine/phi_*.bin" \
  --dynpar-bx "frames_refine/bx_*.bin" --dynpar-by "frames_refine/by_*.bin" \
  --baseline-rho "../014_5_advanced_mhd_fp16_tensorcore/frames/rho_*.bin" \
  --baseline-phi "../014_5_advanced_mhd_fp16_tensorcore/frames/phi_*.bin" \
  --dynpar-shape 128 128 --baseline-shape 192 192 \
  --out dynpar_earth_compare_3d.mp4 --fps 12 --dt 1.0 --stride 2 \
  --tilt-deg 20 --spin-deg 2.5 --line-shells 5 --line-phis 8 --line-scale 0.42
Saved animation to dynpar_earth_compare_3d.mp4 using 120 frames.
```

Notes:
- Set refine_thresh < 0 to disable the dynamic kernel launch.


Day 18: Use Nsight Compute to check occupancy & warp usage; tune block size for best occupancy
Run Nsight Compute across multiple block sizes (run from 018_ncu_occupancy):
```bash
chmod +x run_ncu.sh
./run_ncu.sh 128 128 50 0.15
```

```bash
 ./run_ncu.sh 128 128 50 0.15 > compilation_ncu.txt
ptxas info    : 4 bytes gmem
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 30 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 2.477 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.282 ms
ptxas info    : Compiling entry function '_Z23step_mhd_rusanov_dynparPKfPfiiffff' for 'sm_89'
ptxas info    : Function properties for _Z23step_mhd_rusanov_dynparPKfPfiiffff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 91 registers, used 1 barriers, 392 bytes cmem[0]
ptxas info    : Compile time = 20.202 ms
ptxas info    : Compiling entry function '_Z12refine_patchPKfPfiifffiiii' for 'sm_89'
ptxas info    : Function properties for _Z12refine_patchPKfPfiifffiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 0 barriers, 404 bytes cmem[0]
ptxas info    : Compile time = 17.971 ms
ptxas info    : Compiling entry function '_Z19spectrum_power_binsPK6float2iiPfPji' for 'sm_89'
ptxas info    : Function properties for _Z19spectrum_power_binsPK6float2iiPfPji
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 1.545 ms
ptxas info    : Compiling entry function '_Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i' for 'sm_89'
ptxas info    : Function properties for _Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 1.328 ms
ptxas info    : Compiling entry function '_Z15complex_to_realPfPK6float2if' for 'sm_89'
ptxas info    : Function properties for _Z15complex_to_realPfPK6float2if
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 0.566 ms
ptxas info    : Compiling entry function '_Z15real_to_complexP6float2PKfi' for 'sm_89'
ptxas info    : Function properties for _Z15real_to_complexP6float2PKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.583 ms
ptxas info    : Compiling entry function '_Z14apply_grad_phiPfS_PKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14apply_grad_phiPfS_PKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 396 bytes cmem[0]
ptxas info    : Compile time = 3.259 ms
ptxas info    : Compiling entry function '_Z13subtract_meanPfif' for 'sm_89'
ptxas info    : Function properties for _Z13subtract_meanPfif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 0.503 ms
ptxas info    : Compiling entry function '_Z21apply_poisson_scalingP6float2PKfii' for 'sm_89'
ptxas info    : Function properties for _Z21apply_poisson_scalingP6float2PKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 1.455 ms
ptxas info    : Compiling entry function '_Z7fill_k2Pfii' for 'sm_89'
ptxas info    : Function properties for _Z7fill_k2Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 368 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compile time = 0.767 ms
ptxas info    : 4 bytes gmem
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 30 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 3.351 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.459 ms
ptxas info    : Compiling entry function '_Z23step_mhd_rusanov_dynparPKfPfiiffff' for 'sm_89'
ptxas info    : Function properties for _Z23step_mhd_rusanov_dynparPKfPfiiffff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 91 registers, used 1 barriers, 392 bytes cmem[0]
ptxas info    : Compile time = 20.543 ms
ptxas info    : Compiling entry function '_Z12refine_patchPKfPfiifffiiii' for 'sm_89'
ptxas info    : Function properties for _Z12refine_patchPKfPfiifffiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 0 barriers, 404 bytes cmem[0]
ptxas info    : Compile time = 18.169 ms
ptxas info    : Compiling entry function '_Z19spectrum_power_binsPK6float2iiPfPji' for 'sm_89'
ptxas info    : Function properties for _Z19spectrum_power_binsPK6float2iiPfPji
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 1.699 ms
ptxas info    : Compiling entry function '_Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i' for 'sm_89'
ptxas info    : Function properties for _Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 1.357 ms
ptxas info    : Compiling entry function '_Z15complex_to_realPfPK6float2if' for 'sm_89'
ptxas info    : Function properties for _Z15complex_to_realPfPK6float2if
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 0.640 ms
ptxas info    : Compiling entry function '_Z15real_to_complexP6float2PKfi' for 'sm_89'
ptxas info    : Function properties for _Z15real_to_complexP6float2PKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.508 ms
ptxas info    : Compiling entry function '_Z14apply_grad_phiPfS_PKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14apply_grad_phiPfS_PKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 396 bytes cmem[0]
ptxas info    : Compile time = 3.464 ms
ptxas info    : Compiling entry function '_Z13subtract_meanPfif' for 'sm_89'
ptxas info    : Function properties for _Z13subtract_meanPfif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 0.492 ms
ptxas info    : Compiling entry function '_Z21apply_poisson_scalingP6float2PKfii' for 'sm_89'
ptxas info    : Function properties for _Z21apply_poisson_scalingP6float2PKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 1.340 ms
ptxas info    : Compiling entry function '_Z7fill_k2Pfii' for 'sm_89'
ptxas info    : Function properties for _Z7fill_k2Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 368 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compile time = 1.189 ms
ptxas info    : 4 bytes gmem
ptxas info    : Compiling entry function '_Z15kernel_maxspeedPKfii' for 'sm_89'
ptxas info    : Function properties for _Z15kernel_maxspeedPKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 30 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 2.751 ms
ptxas info    : Compiling entry function '_Z14reset_maxspeedv' for 'sm_89'
ptxas info    : Function properties for _Z14reset_maxspeedv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compile time = 0.334 ms
ptxas info    : Compiling entry function '_Z23step_mhd_rusanov_dynparPKfPfiiffff' for 'sm_89'
ptxas info    : Function properties for _Z23step_mhd_rusanov_dynparPKfPfiiffff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 91 registers, used 1 barriers, 392 bytes cmem[0]
ptxas info    : Compile time = 19.569 ms
ptxas info    : Compiling entry function '_Z12refine_patchPKfPfiifffiiii' for 'sm_89'
ptxas info    : Function properties for _Z12refine_patchPKfPfiifffiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 0 barriers, 404 bytes cmem[0]
ptxas info    : Compile time = 16.558 ms
ptxas info    : Compiling entry function '_Z19spectrum_power_binsPK6float2iiPfPji' for 'sm_89'
ptxas info    : Function properties for _Z19spectrum_power_binsPK6float2iiPfPji
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 1.635 ms
ptxas info    : Compiling entry function '_Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i' for 'sm_89'
ptxas info    : Function properties for _Z22spectrum_make_ke_fieldP6float2PKfS2_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 388 bytes cmem[0]
ptxas info    : Compile time = 1.442 ms
ptxas info    : Compiling entry function '_Z15complex_to_realPfPK6float2if' for 'sm_89'
ptxas info    : Function properties for _Z15complex_to_realPfPK6float2if
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 0.655 ms
ptxas info    : Compiling entry function '_Z15real_to_complexP6float2PKfi' for 'sm_89'
ptxas info    : Function properties for _Z15real_to_complexP6float2PKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.548 ms
ptxas info    : Compiling entry function '_Z14apply_grad_phiPfS_PKfiifff' for 'sm_89'
ptxas info    : Function properties for _Z14apply_grad_phiPfS_PKfiifff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, used 0 barriers, 396 bytes cmem[0]
ptxas info    : Compile time = 3.476 ms
ptxas info    : Compiling entry function '_Z13subtract_meanPfif' for 'sm_89'
ptxas info    : Function properties for _Z13subtract_meanPfif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compile time = 0.528 ms
ptxas info    : Compiling entry function '_Z21apply_poisson_scalingP6float2PKfii' for 'sm_89'
ptxas info    : Function properties for _Z21apply_poisson_scalingP6float2PKfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 376 bytes cmem[0]
ptxas info    : Compile time = 1.261 ms
ptxas info    : Compiling entry function '_Z7fill_k2Pfii' for 'sm_89'
ptxas info    : Function properties for _Z7fill_k2Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 368 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compile time = 0.850 ms
```

Notes:
- The script builds three variants of the Day 17 kernel and runs `ncu --set full`.
- Adjust block sizes by editing `run_ncu.sh`.

Day 19: Install PyTorch 2.2 and test torch.compile on a model; dump generated PTX

Install (CUDA 12.1 wheels are fine on 12.x drivers):
```bash
python -m pip install --upgrade pip
python -m pip install "torch==2.2.*" --index-url https://download.pytorch.org/whl/cu121
```

Run and locate PTX:
```bash
python torch_compile_ptx.py
find torchinductor_cache -name "*.ptx" | head -n 5
```

Notes:
- PTX files land in `torchinductor_cache/` because TORCHINDUCTOR_CACHE_DIR is set in the script.

Day 20: Try TorchRL on CartPole environment; examine an example Triton kernel PTX

Install:
```bash
python -m pip install torchrl gymnasium
```

Run and locate PTX:
```bash
python torchrl_cartpole_ptx.py
find torchinductor_cache -name "*.ptx" | head -n 5
```

Day 21: Use nvdisasm on a Triton kernel SASS; annotate key operations
Dump SASS and annotate (run from 021_nvdisasm_triton):
```bash
chmod +x dump_sass.sh
./dump_sass.sh /path/to/kernel.cubin
```

Notes:
- Find cubins from Day 19/20 with `find ../019_torch_compile_ptx/torchinductor_cache -name "*.cubin"`.
- `annotate_sass.py` tags memory/math/control ops inline.


Day 22: Prototype a Q-learning agent to tune the CFL number in the MHD solver
Build and run:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v mhd_qlearn_cfl.cu -lcufft -o mhd_qlearn_cfl
python qlearn_cfl.py
```

Notes:
- The Q-learning loop calls the solver with different CFL values and rewards low energy drift / no NaNs.

Day 23: Inline PTX atomic add (atom.add.u64) for a custom reward counter
Build and run:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v mhd_atomic_reward.cu -lcufft -o mhd_atomic_reward
./mhd_atomic_reward 128 128 100 --reward-every 10 --rho-min 0.8 --rho-max 1.2 --phi-max 0.2 --b-max 1.5
```

Verify inline PTX:
```bash
nvcc -O3 -arch=sm_89 -ptx mhd_atomic_reward.cu -o mhd_atomic_reward.ptx
grep -n "atom.global.add.u64" mhd_atomic_reward.ptx
```


Day 24:
Read the NVIDIA Hopper (SM90) whitepaper; compare theoretical scheduling model

Day 25: Compile for sm_90 and use cuobjdump to compare SASS vs. sm_86
Folder: 025_sm90_cuobjdump
Run (from the folder):
```bash
./compare_sass.sh
```

Day 26: Install cuTensor; test a small tensor contraction vs. WMMA kernel
Folder: 026_cutensor_wmma
Run (from the folder):
```bash
./run_cutensor_wmma.sh
```

Day 27: Use cuTensorNet for a small RL convolution batch
Folder: 027_cutensornet_conv_rl
Run (from the folder):
```bash
./run_cutensornet_conv.sh
```

Day 28: Nsight Compute L2 residency for conv vs PDE kernel
Folder: 028_ncu_l2_residency
Run (from the folder):
```bash
./profile_l2.sh
```

Day 29: Download Cityscapes (512x) and set up PyTorch dataloader
Folder: 029_cityscapes_dataloader
Run (from the folder):
```bash
./cityscapes_setup.sh
python cityscapes_dataloader.py --split train
```
Notes: Set CITYSCAPES_ROOT or pass --root.

Day 30: Train SegFormer-B0 for one epoch on Cityscapes
Folder: 030_segformer_train
Run (from the folder):
```bash
./run_train.sh
```

Day 31: Export one SegFormer conv layer to PTX via torch.compile
Folder: 031_segformer_ptx
Run (from the folder):
```bash
python segformer_conv_ptx.py
./find_ptx.sh
```

Day 32: Inline PTX to fuse bias + activation in a Triton kernel
Folder: 032_triton_bias_act_ptx
Run (from the folder):
```bash
./run_triton_conv.sh
```

Day 33: CUTLASS 3x3 depthwise conv; inspect SASS
Folder: 033_cutlass_depthwise
Run (from the folder):
```bash
./build_cutlass_depthwise.sh
```
Notes: Set CUTLASS_PATH to a CUTLASS clone if needed.

Day 34: TensorRT 8.6 INT8/FP8 inference; export and inspect
Folder: 034_trt_int8_fp8
Run (from the folder):
```bash
python export_onnx.py
python trt_int8_fp8.py
./inspect_trt_ptx.sh
```

Day 35: Connected-component labeling kernel in PTX
Folder: 035_ptx_cc_label
Run (from the folder):
```bash
./run_cc_label.sh
```

Day 36: Integrate segmentation into CARLA simulation
Folder: 036_carla_segmentation
Run (from the folder):
```bash
./run_carla.sh
```
Notes: Requires CARLA server running on localhost:2000.

Day 37: Clone instant-ngp; render a synthetic MHD volume
Folder: 037_instant_ngp_mhd
Run (from the folder):
```bash
./run_instant_ngp.sh
```

Day 38: Inspect fused NeRF SASS; enable FP16 and re-run
Folder: 038_nerf_sass_fp16
Run (from the folder):
```bash
./build_fp16.sh
./inspect_nerf_sass.sh
```
Notes: Adjust CMake flags if your instant-ngp version differs.

Day 39: Apply INT4 post-training quantization to instant-ngp (proxy)
Folder: 039_instant_ngp_int4
Run (from the folder):
```bash
./run_quant_ngp.sh
```

Day 40: Overlay segmentation masks via CUDA-OpenGL PBO
Folder: 040_cuda_gl_pbo_masks
Run (from the folder):
```bash
./build_pbo.sh
```
Notes: Requires GLFW/GLEW/GL dev packages.

Day 41: Fine-tune 4-bit QLoRA TinyLlama on MHD logs
Folder: 041_qlora_tinyllama
Run (from the folder):
```bash
./run_qlora.sh
```

Day 42: FlashAttention-2; verify ldmatrix PTX
Folder: 042_flashattn2_ldmatrix
Run (from the folder):
```bash
python flashattn2_ldmatrix.py
./inspect_flashattn_ptx.sh
```

Day 43: LangChain RAG on MHD docs
Folder: 043_langchain_rag_mhd
Run (from the folder):
```bash
./run_rag.sh
```

Day 44: Warp-level reduce (inline PTX shuffle) for LoRA
Folder: 044_lora_shuffle_reduce
Run (from the folder):
```bash
./run_lora_shuffle.sh
```

Day 45: Nsight Systems timeline for I/O vs compute overlap
Folder: 045_nsys_io_overlap
Run (from the folder):
```bash
./profile_nsys.sh
```

Day 46: Fuse LoRA forward+backward into CUDA Graph
Folder: 046_lora_cuda_graph
Run (from the folder):
```bash
./run_graph.sh
```

Day 47: Export LoRA model to TensorRT-LLM; benchmark
Folder: 047_trtllm_server
Run (from the folder):
```bash
./trtllm_export.sh
./benchmark_trtllm.sh
```
Notes: Set TRTLLM_PATH and MODEL_PATH.

Day 48: Inspect TensorRT engine PTX for HMMA/DP4A
Folder: 048_trt_engine_ptx
Run (from the folder):
```bash
./inspect_trt_engine.sh /path/to/engine.plan
```

Day 49: Fine-tune TabLLM (4-bit) on magnetics data
Folder: 049_tabllm_finetune
Run (from the folder):
```bash
./run_tabllm.sh
```

Day 50: Measure perplexity vs baseline Llama
Folder: 050_tabllm_perplexity
Run (from the folder):
```bash
python perplexity_eval.py
```

Day 51: FNO for PDE using cuFFT
Folder: 051_fno_cufft
Run (from the folder):
```bash
python fno_pde.py
```

Day 52: Chebyshev dense mat-vec PTX kernel
Folder: 052_chebyshev_ptx
Run (from the folder):
```bash
./run_cheb.sh
```

Day 53: Integrate FNO surrogate into PDE loop
Folder: 053_fno_mhd_surrogate
Run (from the folder):
```bash
python fno_surrogate_mhd.py
```

Day 54: Nsight Compute memory tracing for FNO
Folder: 054_ncu_fno_mem
Run (from the folder):
```bash
./profile_fno_mem.sh
```

Day 55: QAT INT8 for FNO; check accuracy drop
Folder: 055_fno_qat_int8
Run (from the folder):
```bash
python fno_qat_int8.py
```

Day 56: cuQuantum SDK state-vector SVD (4-qubit)
Folder: 056_cuquantum_svd
Run (from the folder):
```bash
python cuquantum_svd.py
```

Day 57: PennyLane VQE on GPU
Folder: 057_pennylane_vqe
Run (from the folder):
```bash
python pennylane_vqe.py
```

Day 58: Inspect PennyLane GPU PTX kernels
Folder: 058_pennylane_ptx
Run (from the folder):
```bash
./inspect_pennylane_ptx.sh
```

Day 59: Use VQE energy as PDE coefficient
Folder: 059_vqe_pde_coupling
Run (from the folder):
```bash
python vqe_pde_coupling.py
```

Day 60: End-to-end GFLOP/s report
Folder: 060_gflops_report
Run (from the folder):
```bash
./run_gflops.sh
```

Day 61: Mesh graph of MHD domain; train PyG GATv2
Folder: 061_pyg_gatv2_mesh
Run (from the folder):
```bash
python mesh_gatv2.py
```

Day 62: Scatter-add kernel with warp shfl.sync (PTX)
Folder: 062_ptx_scatter_shuffle
Run (from the folder):
```bash
./run_scatter.sh
```

Day 63: Train GATv2 for 10 epochs
Folder: 063_gatv2_train
Run (from the folder):
```bash
python train_gatv2.py
```

Day 64: TensorRT conversion of GNN (custom layers)
Folder: 064_trt_gnn_custom
Run (from the folder):
```bash
./run_trtexec.sh
```

Day 65: Insert GNN output as PDE correction
Folder: 065_gnn_pde_correction
Run (from the folder):
```bash
python gnn_pde_correction.py
```

Day 66: CUDA-OpenGL PBO iso-surface render
Folder: 066_cuda_gl_isosurface
Run (from the folder):
```bash
./build_isosurface.sh
```
Notes: Requires GLFW/GLEW/GL dev packages.

Day 67: PTX kernel to copy volume into PBO
Folder: 067_ptx_pbo_copy
Run (from the folder):
```bash
./run_pbo_copy.sh
```

Day 68: Vulkan compute shader PDE update
Folder: 068_vulkan_pde_compute
Run (from the folder):
```bash
./build_spirv.sh
```
Notes: Requires glslangValidator.

Day 69: Compare SPIR-V vs PTX performance
Folder: 069_spirv_ptx_compare
Run (from the folder):
```bash
./compare_spirv_ptx.sh
```

Day 70: EGL headless offscreen rendering
Folder: 070_egl_headless_nerf
Run (from the folder):
```bash
./build_egl.sh
```
Notes: Requires EGL/GLES dev packages.

Day 71: Hopper FP8 Tensor Core dp8a/dp4a demo
Folder: 071_hopper_fp8_dp8a
Run (from the folder):
```bash
./run_dp8a.sh
```

Day 72: SegFormer FP8 GEMM with cuBLASLt
Folder: 072_segformer_fp8_cublaslt
Run (from the folder):
```bash
./run_fp8_gemm.sh
```

Day 73: Profile FP8 kernel metrics
Folder: 073_fp8_ncu_metrics
Run (from the folder):
```bash
./profile_fp8.sh
```

Day 74: DeepSpeed ZeRO-2 on TinyLlama
Folder: 074_deepspeed_zero2
Run (from the folder):
```bash
./run_deepspeed.sh
```

Day 75: Compare iteration time vs QLoRA baseline
Folder: 075_deepspeed_compare
Run (from the folder):
```bash
python compare_iters.py
```

Day 76: CURAND Sobol sequence in device code
Folder: 076_curand_sobol
Run (from the folder):
```bash
./run_sobol.sh
```

Day 77: Embed Sobol noise into solver
Folder: 077_sobol_turbulence
Run (from the folder):
```bash
./run_turbulence.sh
```

Day 78: Vulkan Ray Tracing preview
Folder: 078_vulkan_raytracing
Run (from the folder):
```bash
./build_vkrt.sh
```
Notes: Requires glslangValidator with ray tracing support.

Day 79: DirectX 12 DXR path tracing
Folder: 079_dx12_dxr_nerf
Run (from the folder):
```bash
./compile_dxr.ps1
```
Notes: Run on Windows with dxc.exe available.

Day 80: Enable CUDA MPS; run concurrent workloads
Folder: 080_cuda_mps
Run (from the folder):
```bash
./run_mps.sh
```

Day 81: Train D-NeRF on moving scene
Folder: 081_dnerf_train
Run (from the folder):
```bash
python train_dnerf.py
```

Day 82: INT4 post-training quantization for D-NeRF
Folder: 082_dnerf_int4
Run (from the folder):
```bash
python quantize_dnerf_int4.py
```

Day 83: RL agent to optimize NeRF camera path
Folder: 083_rl_nerf_camera
Run (from the folder):
```bash
python rl_nerf_camera.py
```

Day 84: NCCL loopback multi-process test
Folder: 084_nccl_loopback
Run (from the folder):
```bash
./run_nccl_loopback.sh
```

Day 85: cuobjdump NCCL kernels; HMMA usage
Folder: 085_nccl_cuobjdump
Run (from the folder):
```bash
./inspect_nccl_sass.sh
```

Day 86: OpenACC flux kernel; generate PTX
Folder: 086_openacc_flux
Run (from the folder):
```bash
./build_openacc.sh
```

Day 87: Compare OpenACC vs CUDA runtime
Folder: 087_openacc_vs_cuda
Run (from the folder):
```bash
./compare_openacc_cuda.sh
```

Day 88: cooperative_groups sync vs __syncthreads
Folder: 088_coop_groups_sync
Run (from the folder):
```bash
./run_sync.sh
```

Day 89: Inline bar.sync 0 PTX
Folder: 089_ptx_bar_sync
Run (from the folder):
```bash
./run_bar.sh
```

Day 90: __launch_bounds__ occupancy experiments
Folder: 090_launch_bounds_occ
Run (from the folder):
```bash
./run_launch_bounds.sh
```

Day 91: TensorRT-LLM multi-context server
Folder: 091_trtllm_multicontext
Run (from the folder):
```bash
./start_server.sh
./bench_client.sh
```

Day 92: RLHF loop; reward model PTX
Folder: 092_rlhf_tinyllama
Run (from the folder):
```bash
python rlhf_loop.py
```

Day 93: Collect GPU power metrics
Folder: 093_power_metrics
Run (from the folder):
```bash
./power_metrics.sh
```

Day 94: Wavelet transform on GPU (PTX)
Folder: 094_wavelet_ptx
Run (from the folder):
```bash
./run_wavelet.sh
```

Day 95: CUDA Graph fuse PDE + wavelet
Folder: 095_graph_wavelet_fuse
Run (from the folder):
```bash
./run_graph_wavelet.sh
```

Day 96: Docker container with CUDA/PyTorch/Triton
Folder: 096_docker_stack
Run (from the folder):
```bash
./build_docker.sh
```

Day 97: 6-hour stress test + Nsight Systems
Folder: 097_stress_test_nsys
Run (from the folder):
```bash
./stress_test.sh
```

Day 98: cuTensorNet v2 convolution fusion
Folder: 098_cutensornet_v2_fusion
Run (from the folder):
```bash
python cutensornet_v2_conv.py
```

Day 99: Compare fused vs unfused PTX counts
Folder: 099_nsight_diff_ptx
Run (from the folder):
```bash
./diff_ptx.sh a.ptx b.ptx
```

Day 100: Auto-tuner sweep for block/register settings
Folder: 100_autotuner
Run (from the folder):
```bash
./autotune.sh
```

Day 101: Quantum-inspired simulated annealing PTX kernel
Folder: 101_qi_anneal_ptx
Run (from the folder):
```bash
./run_anneal.sh
```

Day 102: Annealing hyperparameter tuner for MHD (mock fields)
Folder: 102_anneal_tuner_mhd
Run (from the folder):
```bash
python generate_mock_fields.py
python tune_with_anneal.py
```

Day 103: DirectX12 compute PDE update + reference
Folder: 103_dx12_compute_pde
Run (from the folder):
```bash
python generate_mock_input.py
python pde_reference.py
powershell.exe -ExecutionPolicy Bypass -File ./compile_dx12.ps1
```

Day 104: DXIL disassembly vs PTX reference
Folder: 104_dxil_disasm
Run (from the folder):
```bash
./compile_ptx.sh
powershell.exe -ExecutionPolicy Bypass -File ./disasm_dxil.ps1
```

Day 105: NvMedia encode (fallback to NVENC/CPU)
Folder: 105_nvmedia_encode
Run (from the folder):
```bash
python generate_mock_frames.py
./encode_nvmedia.sh
```

Day 106: PTX globaltimer timestamp kernel
Folder: 106_ptx_globaltimer
Run (from the folder):
```bash
./run_globaltimer.sh
```

Day 107: CUDA Graph external dependencies
Folder: 107_graph_ext_deps
Run (from the folder):
```bash
./run_graph_deps.sh
```

Day 108: CUTLASS FP8 GNN GEMM + SASS inspect
Folder: 108_cutlass_fp8_gnn
Run (from the folder):
```bash
./build_cutlass_fp8.sh
./inspect_sass.sh
```

Day 109: Nsight Systems pipeline view workload
Folder: 109_nsys_pipeline
Run (from the folder):
```bash
./profile_pipeline.sh
```

Day 110: TabLLM deployment with TGI + Triton PTX
Folder: 110_tgi_tabllm
Run (from the folder):
```bash
./run_tgi.sh
python test_query.py
./inspect_tgi_ptx.sh
```

Day 111: EGL + NVENC 1080p recording
Folder: 111_egl_nvenc
Run (from the folder):
```bash
python generate_1080p_frames.py
./record_nvenc.sh
```

Day 112: NVENC vs CPU encode throughput
Folder: 112_nvenc_vs_cpu
Run (from the folder):
```bash
./compare_encode.sh
```

Day 113: TensorRT 10 FP4 preview
Folder: 113_trt10_fp4
Run (from the folder):
```bash
./run_trt10_fp4.sh
```

Day 114: Inline PTX dp4a vs HMMA
Folder: 114_dp4a_vs_hmma
Run (from the folder):
```bash
./run_dp4a_vs_hmma.sh
```

Day 115: INT8 DP4A vs FP16 HMMA throughput summary
Folder: 115_int8_fp16_throughput
Run (from the folder):
```bash
python compare_throughput.py
```

Day 116: cuStateVec 10-qubit simulation
Folder: 116_custatevec_10q
Run (from the folder):
```bash
python custatevec_10q.py
```

Day 117: QAOA cost for PDE parameters
Folder: 117_qaoa_pde
Run (from the folder):
```bash
python qaoa_pde.py
```

Day 118: QAOA gradient PTX export
Folder: 118_qaoa_grad_ptx
Run (from the folder):
```bash
python qaoa_grad_ptx.py
./find_ptx.sh
```

Day 119: ldmatrix-style warp conv kernel
Folder: 119_ldmatrix_conv
Run (from the folder):
```bash
./build_ldmatrix.sh
```

Day 120: Launch bounds on conv kernel
Folder: 120_launch_bounds_conv
Run (from the folder):
```bash
./run_launch_bounds.sh
```

Day 121: Fourier feature MLP + PTX check
Folder: 121_fourier_mlp
Run (from the folder):
```bash
python generate_mock_inputs.py
python fourier_mlp.py
./find_ptx.sh
```

Day 122: FP8 WMMA triple-buffering demo
Folder: 122_fp8_triplebuf
Run (from the folder):
```bash
./run_fp8_triplebuf.sh
```

Day 123: Nsight occupancy heatmap
Folder: 123_ncu_occ_heatmap
Run (from the folder):
```bash
./profile_occ_heatmap.sh
```

Day 124: Cooperative kernel multi-stream overlap
Folder: 124_coop_multistream
Run (from the folder):
```bash
./run_coop.sh
```

Day 125: L2/texture cache metrics on sampling
Folder: 125_nerf_cache_metrics
Run (from the folder):
```bash
./profile_cache.sh
```

Day 126: Philox vs XORShift RNG distribution
Folder: 126_philox_vs_xorshift
Run (from the folder):
```bash
./run_rng_compare.sh
```

Day 127: RL noise stability test
Folder: 127_rl_noise_stability
Run (from the folder):
```bash
python rl_noise.py
```

Day 128: Vulkan async compute overlap
Folder: 128_vulkan_async_overlap
Run (from the folder):
```bash
./build_async.sh
```

Day 129: AMD RGP trace capture
Folder: 129_amd_rgp_trace
Run (from the folder):
```bash
./run_rgp.sh
```

Day 130: Nsight stall ratio summary
Folder: 130_nsight_stall_ratio
Run (from the folder):
```bash
./collect_stalls.sh
python stall_ratio.py
```

Day 131: DeepSpeed ZeRO-3 offload
Folder: 131_ds_zero3_offload
Run (from the folder):
```bash
./run_zero3.sh
```

Day 132: NSYS transfer overlap trace
Folder: 132_nsys_nvlink_overlap
Run (from the folder):
```bash
./profile_transfers.sh
```

Day 133: Triton reduction with inline asm
Folder: 133_triton_reduction_asm
Run (from the folder):
```bash
./run_triton_reduce.sh
```

Day 134: Triton vs CUB register usage
Folder: 134_triton_vs_cub_regs
Run (from the folder):
```bash
./compare_regs.sh
```

Day 135: FNO gradient checkpointing
Folder: 135_fno_checkpointing
Run (from the folder):
```bash
python fno_checkpoint.py
```

Day 136: VRAM profiling during training
Folder: 136_vram_profile
Run (from the folder):
```bash
./profile_vram.sh
```

Day 137: GPT-J INT4 inference test
Folder: 137_gptj_int4
Run (from the folder):
```bash
./run_gptj_int4.sh
```

Day 138: GPT-J vs TinyLlama latency
Folder: 138_gptj_vs_tinyllama_latency
Run (from the folder):
```bash
python compare_latency.py
```

Day 139: cuTensor 4th-order contraction
Folder: 139_cutensor_4th_order
Run (from the folder):
```bash
./run_cutensor_4d.sh
```

Day 140: CUTLASS experimental FFT vs cuFFT
Folder: 140_cutlass_fft
Run (from the folder):
```bash
./run_cutlass_fft.sh
```

Day 141: SM90 microcode scheduling notes + occupancy
Folder: 141_sm90_microcode_notes
Run (from the folder):
```bash
./collect_occupancy.sh
python summarize_microcode.py
```

Day 142: Instruction reorder SASS experiment
Folder: 142_sass_reorder_test
Run (from the folder):
```bash
./run_reorder.sh
```

Day 143: Nsight source stall analysis
Folder: 143_ncu_source_stalls
Run (from the folder):
```bash
./profile_stalls.sh
```

Day 144: Warp-level 3x3 cross product PTX
Folder: 144_ptx_cross_product
Run (from the folder):
```bash
./run_cross.sh
```

Day 145: Cross product vs WMMA comparison
Folder: 145_cross_vs_wmma
Run (from the folder):
```bash
./run_cross_vs_wmma.sh
```

Day 146: Occupancy auto-tune sweep
Folder: 146_occ_autotune
Run (from the folder):
```bash
./run_autotune.sh
```

Day 147: Shared memory bank conflict sweep
Folder: 147_reg_bank_conflicts
Run (from the folder):
```bash
./run_bank_conflicts.sh
```

Day 148: Flux kernel optimized vs baseline
Folder: 148_flux_kernel_finalize
Run (from the folder):
```bash
./run_flux_bench.sh
```

Day 149: Speedup documentation summary
Folder: 149_speedup_doc
Run (from the folder):
```bash
./run_speedup_doc.sh
```

Day 150: Full pipeline integration test
Folder: 150_full_pipeline_test
Run (from the folder):
```bash
./run_full_pipeline.sh
```

Day 151: CURAND Sobol device vs host quality
Folder: 151_curand_sobol_quality
Run (from the folder):
```bash
./run_sobol_compare.sh
```

Day 152: Warp-level XORSHIFT PTX RNG
Folder: 152_ptx_xorshift_warp
Run (from the folder):
```bash
./run_xorshift.sh
```

Day 153: RNG quality comparison (chi-square/KS)
Folder: 153_rng_quality_compare
Run (from the folder):
```bash
./run_rng_quality.sh
```

Day 154: RL random resets perturbation
Folder: 154_rl_random_resets
Run (from the folder):
```bash
./run_random_resets.sh
```

Day 155: TensorBoard reward stability logs
Folder: 155_tensorboard_stability
Run (from the folder):
```bash
./run_tensorboard.sh
```

Day 156: A-Trous denoiser (Vulkan shader + CPU)
Folder: 156_vulkan_atrous_denoise
Run (from the folder):
```bash
./run_atrous.sh
```

Day 157: PTX wavelet transform verification
Folder: 157_ptx_wavelet_denoise
Run (from the folder):
```bash
./run_wavelet_ptx.sh
```

Day 158: Denoiser timing benchmark
Folder: 158_denoise_timing
Run (from the folder):
```bash
./run_denoise_timing.sh
```

Day 159: Denoiser shared memory bank tuning
Folder: 159_denoise_bank_tuning
Run (from the folder):
```bash
./run_bank_tune.sh
```

Day 160: Denoiser quality comparison
Folder: 160_denoise_quality_compare
Run (from the folder):
```bash
./run_quality_compare.sh
```

Day 161: GNN flux Jacobian surrogate
Folder: 161_gnn_flux_surrogate
Run (from the folder):
```bash
./run_gnn_train.sh
```

Day 162: GNN TorchInductor PTX export
Folder: 162_gnn_inductor_ptx
Run (from the folder):
```bash
python gnn_inductor_ptx.py
./find_ptx.sh
```

Day 163: GNN fused kernel SASS inspect
Folder: 163_gnn_sass_inspect
Run (from the folder):
```bash
./inspect_gnn_sass.sh
```

Day 164: cuTensorNet v2 fusion mock benchmark
Folder: 164_cutensornet_fusion
Run (from the folder):
```bash
./run_fusion.sh
```

Day 165: cuTensorNet fused vs separate timing
Folder: 165_cutensornet_timing
Run (from the folder):
```bash
./run_fusion_timing.sh
```

Day 166: FP8 (dp4a) conv pipeline demo
Folder: 166_fp8_conv_dp8a
Run (from the folder):
```bash
./run_fp8_conv.sh
```

Day 167: Nsight power/clock logging
Folder: 167_nsight_power_clock
Run (from the folder):
```bash
./run_power_profile.sh
```

Day 168: Thermal throttling sweep
Folder: 168_thermal_throttle_log
Run (from the folder):
```bash
./run_thermal_sweep.sh
```

Day 169: CUDA 13.1 release notes check
Folder: 169_cuda_131_release
Run (from the folder):
```bash
./check_cuda_131.sh
```

Day 170: CUDA 13.1 benchmark rerun
Folder: 170_cuda_131_benchmarks
Run (from the folder):
```bash
./run_cuda_131_benchmarks.sh
```

Day 171: OpenACC offload sample (NVC++)
Folder: 171_openacc_cutensor
Run (from the folder):
```bash
./build_openacc.sh
```

Day 172: OpenACC vs nvcc PTX compare
Folder: 172_openacc_vs_nvcc_ptx
Run (from the folder):
```bash
./compare_ptx.sh
```

Day 173: Domain split NVLink simulation
Folder: 173_nvlink_domain_split
Run (from the folder):
```bash
./run_domain_split.sh
```

Day 174: MPS partitioned CUDA Graph
Folder: 174_mps_graph_partition
Run (from the folder):
```bash
./run_mps_graph.sh
```

Day 175: Nsight Systems MPS overlap
Folder: 175_nsys_mps_overlap
Run (from the folder):
```bash
./profile_mps_nsys.sh
```

Day 176: gRPC RL parameter server
Folder: 176_grpc_param_server
Run (from the folder):
```bash
./run_grpc.sh
```

Day 177: CPU-GPU launch latency
Folder: 177_cpu_gpu_latency
Run (from the folder):
```bash
./run_latency.sh
```

Day 178: Async memcpy overlap
Folder: 178_async_memcpy_overlap
Run (from the folder):
```bash
./run_async_copy.sh
```

Day 179: L2 hit ratio profiling
Folder: 179_l2_hit_overlap
Run (from the folder):
```bash
./profile_l2.sh
```

Day 180: Kernel speed report
Folder: 180_speed_report
Run (from the folder):
```bash
./run_speed_report.sh
```

Day 181: DX12 DXR NeRF mock pipeline
Folder: 181_dx12_dxr_nerf
Run (from the folder):
```bash
./run_dxr.sh
```

Day 182: GL_NV_copy_image mock copy
Folder: 182_gl_nv_copy_image
Run (from the folder):
```bash
./run_gl_copy.sh
```

Day 183: Direct vs staging copy compare
Folder: 183_copy_vs_staging
Run (from the folder):
```bash
./run_copy_compare.sh
```

Day 184: ImGui HUD overlay mock
Folder: 184_imgui_hud_overlay
Run (from the folder):
```bash
./run_hud_overlay.sh
```

Day 185: RL HUD sensor fusion
Folder: 185_rl_hud_fusion
Run (from the folder):
```bash
./run_hud_fusion.sh
```

Day 186: RL success rate comparison
Folder: 186_rl_success_rate
Run (from the folder):
```bash
./run_success_rate.sh
```

Day 187: SegFormer FP4 mock quant
Folder: 187_segformer_fp4_trt
Run (from the folder):
```bash
./run_fp4_quantize.sh
```

Day 188: TensorRT FP4 PTX inspect
Folder: 188_trt_engine_fp4_ptx
Run (from the folder):
```bash
./inspect_trt_fp4_ptx.sh
```

Day 189: SegFormer FP4 accuracy drop
Folder: 189_segformer_fp4_accuracy
Run (from the folder):
```bash
./run_accuracy.sh
```

Day 190: LoRA recovery on FP4
Folder: 190_lora_fp4_recover
Run (from the folder):
```bash
./run_lora_recover.sh
```

Day 191: qGAN PennyLane demo
Folder: 191_qgan_pennylane
Run (from the folder):
```bash
./run_qgan.sh
```

Day 192: qGAN cuTensorNet conv
Folder: 192_qgan_cutensornet
Run (from the folder):
```bash
./run_cutensornet_qgan.sh
```

Day 193: qGAN conv SASS inspect
Folder: 193_qgan_sass
Run (from the folder):
```bash
./inspect_qgan_sass.sh
```

Day 194: qGAN RL domain randomization
Folder: 194_qgan_rl_domain
Run (from the folder):
```bash
./run_qgan_rl.sh
```

Day 195: qGAN FID score
Folder: 195_qgan_fid
Run (from the folder):
```bash
./run_fid.sh
```

Day 196: Multi-stream CUDA Graph events
Folder: 196_graph_multistream_events
Run (from the folder):
```bash
./run_multigraph.sh
```

Day 197: cudaEventBlockingSync overhead
Folder: 197_event_blocking_overhead
Run (from the folder):
```bash
./run_event_overhead.sh
```

Day 198: Nsight overlap with events
Folder: 198_nsys_overlap_events
Run (from the folder):
```bash
./profile_overlap.sh
```

Day 199: Chunk size tuning
Folder: 199_chunk_size_tuning
Run (from the folder):
```bash
./run_chunk_tuning.sh
```

Day 200: 8-hour pipeline stress test
Folder: 200_stress_test_8h
Run (from the folder):
```bash
./run_stress_8h.sh 8
```

Day 201:
Build a custom convolution kernel in Triton 2.2 using new `asm` macros; verify PTX matches expectations 

Day 202:
Add a group synchronization test with PTX mbarrier (new in 8.0); experiment with producer-consumer in one kernel 

Day 203:
Evaluate how the mbarrier affects compute/latency pipeline in a microbenchmark
 
Day 204:
Integrate the INT4 GNN + PDE + NeRF inference into one pipeline (multi-model inference); ensure fits in 16 GB

Day 205:
Measure VRAM residency of each model when running concurrently (use nvidia-smi continuously)

Day 206:
Use the CUDA Memory Pool API (`cudaMemPool_t`) to manage allocations and reduce fragmentation 

Day 207:
Observe if memory fragmentation is reduced by using a custom memory pool (log memory usage over time)

Day 208:
Add dynamic parallelism for adaptive mesh refinement: launch sub-kernels for fine grid only where needed

Day 209:
Compare the overhead of nested kernel launches vs. doing same work on host (time a step with vs. without adaptivity)

Day 210:
Write notes on nested kernel occupancy and limitations (for documentation) 

Day 211:
Train a depth-aware NeRF (with DepthFusion technique) to incorporate LiDAR-like depth supervision

Day 212:
Use pseudo-LiDAR depth maps as additional input to NeRF training (fusion of image+depth)

Day 213:
Write a PTX gather kernel to fuse depth information into NeRF’s radiance field update 

Day 214:
Compare PSNR of rendered views with depth supervision vs. without; see improvement

Day 215:
Let the RL agent use the depth error (difference between predicted vs. true depth) as a penalty in reward; fine-tune agent

Day 216:
Update to TensorRT-LLM v1.1; use new fused MHA kernels for LLM inference; measure throughput gain

Day 217:
Dump the new TensorRT fused kernel SASS; look for `HFMA2.MMA` (`Hopper FP8 FMA`) usage 

Day 218:
Evaluate prompt latency with TRT-LLM v1.1 vs. v1.0 on a 13B model; log any improvement

Day 219:
Add retrieval-augmentation to LLM inference using FAISS (vector DB) for MHD documents

Day 220:
Measure answer recall/accuracy with RAG vs. without (using a test set of questions)

Day 221:
Launch multiple GPU streams with copy engine saturation test: e.g., 4 streams each doing copies and compute

Day 222:
Use cudaMemcpyAsync and `cudaGraphAddMemcpyNode` to schedule async transfers in a CUDA Graph

Day 223:
Use Nsight Systems to trace copy vs. compute overlap; verify copy engine utilization is near 100%

Day 224:
Tune the chunk sizes of memcpy to maximize overlap and throughput (empirically find sweet spot)

Day 225:
Try the cuDNN Frontend API to auto-build a fused `conv+bias+activation`; compare to hand-written version 

Day 226:
Inspect the PTX from the cuDNN-generated fused kernel vs. our manual one; see differences

Day 227:
Insert an inline PTX activation (e.g., HMMA with RELU in epilogue) to the custom conv to mimic cuDNN epilogue fusion

Day 228:
Compare the speed of the kernel with manual epilogue vs. separate activation call; log improvement

Day 229:
Apply QAT INT8 to the conv (simulate training quantized); ensure Tensor Cores (DP4A) are used by checking PTX

Day 230:
Evaluate the accuracy of the INT8 model vs. FP32 on a validation set; ensure drop is within acceptable range

Day 231:
Install CUDA 13.2 Beta if available; look at any new PTX features (maybe PTX 5.2)

Day 232:
Recompile key kernels under CUDA 13.2; note the PTX version and any changes in SASS

Day 233:
Read up on the new Tensor Memory Accelerator (TMA) in Hopper; understand how it can improve memory copy

Day 234:
Prototype using cp.async.bulk PTX (if supported) for large bulk copy operations in a kernel

Day 235:
Compare traditional global memory copy vs. TMA approach (simulate if hardware not available)

Day 236:
Implement Fourier feature look-ups (embedding) via TMA in a kernel, if possible; measure L2 hit reduction

Day 237:
Measure if using TMA for memory brings L2 hit rate down (meaning less thrashing); use Nsight metrics 

Day 238:
Add an FP8 dropout kernel with inline PTX (random bitmask applied to FP8 tensor) in training; test stability

Day 239:
Check training stability/accuracy when using FP8 with dropout vs. FP16; log any divergence issues

Day 240:
Take a snapshot of the entire project repository; ensure all code/notebooks are saved (e.g., commit to Git) 

Day 241:
Dive deep into instruction scheduling: use Nsight’s instruction timeline view for a kernel to mark latency per instruction group

Day 242:
Annotate the timeline with pipeline stages (memory vs ALU) for better understanding of bottlenecks 

Day 243:
Try reordering instructions in a kernel (e.g., move load instructions earlier) to hide latency; re-profile IPC

Day 244:
Measure IPC (instructions per cycle) difference after reordering; verify if warp stall count reduced 

Day 245:
Study register file bank conflicts from an NVIDIA doc; plan an experiment to intentionally cause bank conflicts 

Day 246:
Modify a kernel to force register bank conflict (if possible) and see if Nsight reports increased stall or lower occupancy

Day 247:
Use `__syncwarp()` (new warp sync) versus `bar.sync` to synchronize within warp; measure any difference in overhead

Day 248:
Use C++17 parallel STL with `<execution>` (NVC++ stdpar) to offload a simple loop to GPU; compare PTX vs. explicit CUDA

Day 249:
Compare the PTX from stdpar compiled code to our earlier hand-written kernel; see if optimizations differ 

Day 250:
Run a full-system 24-hour soak test on the pipeline; log GPU temperatures, clock speeds, and any errors

Day 251:
Install the preview of cuTe (Tensor Extensions) if available; port one convolution to cuTe; compare PTX

Day 252:
Compare the PTX or performance of the cuTe convolution vs. CUTLASS baseline; note differences

Day 253:
Add an RLHF reward model kernel in PTX to the pipeline (simulate inference of a small reward model); test integration

Day 254:
Train the RLHF policy for 2 epochs on TinyLlama outputs; monitor VRAM usage and throughput

Day 255:
Integrate the RLHF-tuned policy into the RL agent in the environment; test agent performance improvement

Day 256:
Use Nsight Systems for a holistic view of GPU, CPU, I/O across the pipeline after RLHF integration

Day 257:
Adjust thread priorities of encoding vs. rendering threads (e.g., use `chrt` on Linux) to optimize pipeline flow

Day 258: Re-tune the GPU power limit for sustained performance (lower if thermal throttling); log sustained clocks

Day 259: After 30 minutes at new power limit, evaluate if GPU clocks remain stable vs. before (log analysis) 

Day 260: Evaluate long-run stability: no memory leaks, and performance does not degrade over 30 min+ runs 

Day 261: Test CUDA-Aware MPI (OpenMPI) inside WSL: send/recv GPU buffers directly; measure latency vs. CPU buffer 

Day 262: Use MPI to split the PDE domain across 2 processes (on one GPU via time-slicing); measure overhead of communication

Day 263: Compare direct GPU-to-GPU P2P copy vs. staging through host for these MPI messages (simulate using cudaMemcpy) 

Day 264: Implement a novel FP6 quantization (NetQRE) for LLM weights; integrate into TinyLlama and test perplexity 

Day 265: Inspect PTX for FP6 operations or how it emulates (likely using INT8 with scale); see if any new instructions 

Day 266: Evaluate the accuracy of FP6 quantized model vs. FP8 and FP4; see if it offers a middle ground in quality 

Day 267: Experiment with offloading the LLM KV-cache to disk when GPU memory is exceeded (using paging or streaming) 

Day 268: Measure hit/miss rates and latency impact of disk-paged KV cache (simulate by artificially small cache on GPU) 

Day 269: Tune NVENC encoder preset (quality vs. speed) to find best balance for live streaming the output; test latency impact 

Day 270: Plot the latency vs. quality trade-off for different NVENC presets or bitrates; identify optimal point 

Day 271: Build a stdpar version of the PDE solver (C++ parallel STL); compile with NVC++ and run on GPU 

Day 272: Inspect the generated PTX from stdpar PDE vs. our original CUDA PDE; ensure it vectorized loads properly 

Day 273: Compare runtime of stdpar solver vs. hand-optimized CUDA solver; measure difference in GFLOP/s 

Day 274: Add cp.async multi-stage copies for boundary halo exchange in PDE (overlap communication in shared mem) 

Day 275: Use mbarrier and async.commit_group to coordinate the multi-stage copy (Hopper feature); test on sm_90 if possible

Day 276: Validate that results remain the same with the new asynchronous copy scheme (bitwise compare outputs) 

Day 277: Integrate a mixed-precision strategy (FP8 for some layers, FP16 for others) in conv layers of the ML models; test if training converges 

Day 278: Check PTX of mixed precision kernels to see both FP16 and FP8 ops; confirm scheduler alternates them effectively 

Day 279: Evaluate throughput improvement with mixed precision vs. pure FP16; chart the differences 

Day 280: Push the entire project code to a public GitHub repository (if possible); ensure no proprietary data 

Day 281: Install Triton 3.0 Alpha (if available); port one of our custom Triton kernels to it; examine changes/new scheduler 

Day 282: Measure performance of the Triton 3.0 kernel vs. Triton 2.x version; note any improvements or regressions 

Day 283: Implement a block-sparse attention or transformer with block-sparse kernels (e.g., using Sparse GPT code) 

Day 284: Use CUTLASS or PyTorch SPMM to accelerate the block-sparse attention; examine SASS (should use optimized sparse Tensor Core) 

Day 285: Evaluate memory savings and speed of block-sparse model vs. dense baseline; log results 

Day 286: Add a curriculum learning schedule to the RL training (start easy, progressively harder tasks) 

Day 287: Code a custom reward shaping kernel in PTX (to combine multiple reward signals); integrate into training loop 

Day 288: Monitor the learning curve with curriculum and shaped rewards vs. original; verify faster convergence 

Day 289: Perform a 6-hour pipeline stress test after all optimizations; track VRAM usage to catch leaks (loop nvidia-smi) 

Day 290: Fix any memory leaks or issues found; retest to ensure stability in long runs

Day 291: Connect an AutoGPT or LLM agent to the pipeline for auto-tuning: let it propose hyperparameters based on logs 

Day 292: Have the agent generate candidate configurations; evaluate PDE error or reward for each; let it iterate 

Day 293: Compare kernels or parameters chosen by the AI agent vs. our manual best; analyze any novel strategies it found 

Day 294: Integrate a citation extraction NLP (e.g., CITEX) to scan new papers (like magnetohydrodynamics) for relevant formulas/ideas 

Day 295: Quantize the CITEX model to 4-bit and deploy it with TGI for quick Q&A on new papers (like an AI researcher assistant) 

Day 296: Full-Day Integration Test: Run a 12-hour continuous test with all modules (PDE+NeRF+RL+LLM+CV) active; monitor and log everything 

Day 297: Use Nsight Systems to capture a full timeline (.qdrep) of the 12h run; save for analysis 

Day 298: Summarize overall throughput: GFLOP/s for simulation, FPS for rendering, tokens/s for LLM, etc., from the run logs 

Day 299: Perform system maintenance: clean temporary files, verify drivers, possibly reinstall NVIDIA driver to refresh state 

Day 300: Grand 24h Stress & Benchmark: Run the entire pipeline for 24 hours straight; log any failures, final performance metrics, and archive results

Day 301: Satellite Imagery Segmentation Project Begins - Prepare dataset of satellite images (optical and SAR) with building outlines (e.g., SpaceNet or OpenEarthMap-SAR)

Day 302: Pretrain a Masked Autoencoder (MAE) on the satellite images (self-supervised) to learn representations

Day 303:Fine-tune a segmentation model (e.g., U-Net or SegFormer) using the MAE encoder on building outlines; aim for high Dice coefficient 

Day 304:Incorporate SAR channel data alongside optical images in training; observe if segmentation accuracy (Dice) improves (expect ~10% $\uparrow$) 

Day 305:Evaluate the model on a test region; if possible, overlay predicted outlines on imagery to visually check results 

Day 306: Parse news articles (text/images) for mentions of new buildings (simulate with sample data); use a language model to extract location info

Day 307: Use a Retrieval-Augmented Generation (RAG) approach with an LLM (e.g., Llama 2) to add newly mentioned buildings into the existing map database

Day 308: Real-Time QEC Decoder Project Begins - Set up NVIDIA CUDA Quantum (CUDA-Q) environment; run a basic hybrid quantum-classical example

Day 309: Simulate a small quantum error correcting code (e.g., 3-qubit bit-flip code) using cuQuantum; generate error syndromes

Day 310: Implement a baseline CPU decoder (e.g., brute force or simple lookup) for the code; measure its latency per round

Day 311: Train a small Transformer or GNN-based decoder on simulated syndrome error data; use PyTorch on GPU for training

Day 312: Deploy the learned decoder model on GPU (inference); compare its latency to CPU baseline (expect ~2-3x speedup)

Day 313: Integrate the GPU decoder into a loop simulating real-time error correction cycles; ensure it keeps up with the error rate (e.g., <1 ms per round)

Day 314: Read the Ring Attention paper (for near-infinite context LLMs); understand how it distributes attention across GPUs

Day 315: Simulate Ring Attention on a single GPU by splitting a long sequence into blocks; write code to process blocks sequentially (mimicking multi-GPU)

Day 316: Measure memory usage and time for processing a long sequence with our blockwise approach vs. naive full attention ($O(N^2)$); extrapolate benefits

Day 317: Read about Striped Attention (alternative to Ring); compare their approaches (contiguous vs. interleaved token partitioning)

Day 318: Consider how to implement Striped Attention on one GPU (alternating tokens); code a simple prototype and compare computational load per step

Day 319: Triton Kernel Optimization - Write a Triton kernel for INT4 matrix multiplication (simulate dequantization + matmul as in GPTQ)

Day 320: Optimize the Triton INT4 kernel: coalesce memory accesses, use appropriate tile sizes, unroll loops; aim to surpass initial throughput by 2-3x

Day 321: Profile the optimized Triton kernel with Nsight Compute; ensure higher occupancy and less warp stall (compare metrics before vs. after)

Day 322: FastVideo Project Exploration - Read FastVideo's architecture overview (modular video diffusion pipeline, optimizations like sparse attention)

Day 323: Run a FastVideo provided example (video generation with Wan2.1 model) on a short prompt; measure generation speed (frames/sec)

Day 324: Examine FastVideo's optimized attention (e.g., Sliding Tile Attention); read the corresponding paper to grasp how it works

Day 325: Dive into FastVideo code (e.g., fastvideo/attention/): find a Triton or CUDA kernel implementation for attention; note any unique tricks

Day 326: Apple Metal GPU Programming - (if Mac accessible) Read Apple Metal compute shader documentation; understand kernel execution model

Day 327: Write a simple Metal compute kernel (e.g., vector add) and run on an Apple Silicon GPU (or simply write and compile if no device); verify correct result

Day 328: TensorRT Deployment - Write a C++ (or Python) program using TensorRT APIs to load a small ONNX model and run inference; measure latency

Day 329: Implement a custom TensorRT plugin layer (e.g., a non-standard activation); integrate it into an engine and test that it works and improves performance

Day 330: (Optional) Build PyTorch from source and run a simple test to ensure familiarity with its internals; or attempt a small modification (like print in an op) to see build process

Day 331: AMD GPU Insight - Read about AMD's Composable Kernel (CK) library in MIOpen (how templated kernels achieve performance)

Day 332: If an AMD GPU is available (or using ROCm on CPU), try to compile and run a small kernel with the Composable Kernel library (e.g., a GEMM); otherwise, analyze provided CK examples for structure

Day 333: Implement structured sparsity (NVIDIA Ampere 2:4) in a layer: prune a dense weight matrix to 50% 2:4 sparse, then use cuSPARSELt to execute a sparse GEMM; compare speed to dense GEMM

Day 334: Try Apache TVM: use it to compile a simple model (e.g., ResNet-18 or a single layer) for CUDA; run the model and compare inference speed to PyTorch's native run

Day 335: Compare multi-framework performance: take a small model and benchmark it under PyTorch (eager and torch.compile), TensorRT, and TVM; summarize latency and throughput for each

Day 336: Determine the break-even point for GPU vs. CPU: run a simple operation (like a small matrix multiply or sum) with varying sizes on GPU and CPU to find when GPU becomes faster despite launch overhead

Day 337: Investigate Python overhead: measure the cost of launching many small CUDA kernels vs. doing the equivalent work in one larger kernel (e.g., summing an array in chunks vs. whole); use events or timers to quantify overhead

Day 338: Experiment with Mirage (GPU kernel superoptimizer): define a simple attention computation in Mirage and generate an optimized kernel; compare its performance to our Triton or PyTorch kernel for the same operation

Day 339: Test JAX on GPU: implement a small model (or use a JAX example) and compare its performance to PyTorch (with/without `torch.compile`) on the same task; observe differences in compilation time and speed

Day 340: Measure how batch size affects throughput: take an inference task (like SegFormer or ResNet) and benchmark inference latency for batch = 1, 2, 4, 8, ...; plot throughput (images/sec) vs. batch to find optimal range

Day 341: Measure how sequence length affects LLM throughput: use a model (like GPT-2 or Llama) to generate with context lengths 512, 1024, 2048, etc.; record tokens/sec vs. sequence length to illustrate $O(N^2)$ effect 

Day 342: Demonstrate effect of KV caching: generate text with an LLM twice - once with caching (only new tokens attended) and once without (recompute attention on full sequence each step); measure the speed difference per token

Day 343: Simulate concurrent inference: launch two processes (or threads) each running an inference (e.g., two different models or same model with different inputs) on the GPU; observe if total throughput increases (utilizing idle periods)

Day 344: Enable CUDA MPS and repeat the concurrent inference test; measure latency and throughput when MPS is coordinating multiple contexts vs. without MPS (time-slicing)

Day 345: Test mixed precision vs. full precision: run a representative inference (e.g., ResNet or transformer) in FP32, FP16, and INT8 (using quantized model) and measure latency for each; confirm Tensor Cores usage via profiling for FP16/INT8

Day 346: Use NVIDIA DALI to accelerate data preprocessing: replace a CPU data loader (for image augmentation or video frames) with a DALI pipeline on GPU; measure end-to-end training or inference speed improvement

Day 347: Plot and analyze results: generate a graph of LLM tokens/sec vs. context length (from day 341 data) to visualize the scaling problem; ensure the plot clearly shows how longer context reduces throughput

Day 348: Plot batch size vs. throughput using data from day 340; identify where diminishing returns set in for batching on the GPU

Day 349: Plot FP32 vs FP16 vs INT8 latency from day 345 to illustrate speedup from lower precision; annotate any accuracy differences observed

Day 350: Visualize segmentation output: take a sample satellite image and overlay the model's building outline predictions in color; save this as an image to verify qualitative results

Day 351: Export the trained satellite segmentation model for deployment: use torch.export to get a stable graph, convert to ONNX, and run it through TensorRT; compare inference speed on a test image vs. PyTorch

Day 352: Deploy the QEC decoder model: if it's small enough, convert it to TensorRT or a custom CUDA kernel (since it might be just MLP/attention); ensure it runs within the required latency per cycle

Day 353: Create a final consolidated report of improvements: list each major optimization (quantization, sparsity, new kernels, etc.) and the achieved speedups or memory savings in our projects (MHD, NeRF, LLM, CV)

Day 354: Prepare a brief presentation (slides or document) summarizing the entire 365-day project: include key findings, performance numbers, and future work ideas

Day 355: (Buffer day) Address any remaining issues or TODOs that came up during the projects (e.g., unfinished experiments, additional tuning suggested by results)

Day 356: Explore any new release or tool announced recently (e.g., a new version of PyTorch, CUDA, or a novel model optimization technique in late 2025) and consider how it might apply to our pipeline

Day 357: Final code refactoring: clean up and refactor the codebase for readability, modularity, and prepare it for possible public release (if applicable)

Day 358: Run an end-to-end test of the refactored code to ensure everything still works and is reproducible from scratch (scripts, README, etc.)

Day 359: Document the environment and dependencies (versions of CUDA, drivers, libraries) so that the project can be set up elsewhere if needed

Day 360: Engage in a code review or self-review: examine critical sections (PTX code, custom kernels) to ensure they are correct and optimal; add comments where necessary for future maintainers

Day 361: Prepare for deployment or next steps: if this project were to continue, outline a plan for turning it into a real product or research paper (e.g., what pieces need further development)

Day 362: Knowledge sharing: write a blog post or internal wiki article detailing the journey, the technical challenges overcome, and key learnings for colleagues or the community

Day 363: Presentation day: present the results of the projects to peers or mentors, including demos of the running pipeline (segmentation, simulation, RL, LLM responding) and Q&A

Day 364: Revise CUDA, Triton, RTX, ML frameworks, etc

Day 365: 
