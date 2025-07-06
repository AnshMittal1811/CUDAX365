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


Day 13:
Write a 32×32 butterfly FFT kernel in PTX; compare its runtime with cuFFT 

Day 14:
Switch MHD solver to FP16 + Tensor Cores; verify HMMA instructions in PTX

Day 15:
Inline approximate reciprocal (`rcp.approx.f32`) PTX in the flux loop to replace division 

Extras: 
Basic GitHub project structure for your HPC experiments.
Reference: Git Book - Best Practices

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

---
Block 4 (Days 16–20)

Day 16:
Start exploring Quantum Computing basics: Qubits, gates, Bloch sphere, from a coding perspective.
Reference: Qiskit Textbook

Day 17:
Install Qiskit or Cirq on WSL2. Run basic quantum circuits on the local simulator.
Reference: Qiskit Installation Guide

Day 18:
Implement a simple quantum algorithm (e.g., quantum teleportation or Grover’s) in Qiskit.
Reference: Grover’s Algorithm Qiskit Tutorial

Day 19:
Introduction to Quantum Machine Learning: conceptual overview of parameterized quantum circuits.
Reference: PennyLane tutorials

Day 20:
Try a small QML example (variational circuit) with PennyLane or Qiskit Machine Learning.
Reference: Qiskit Machine Learning Docs

---
Block 5 (Days 21–25)
Day 21:
Introduction to Quantum Chromodynamics (QCD) and how ML is used (e.g., lattice QCD data analysis).
Reference: Lectures: “Machine Learning for Quantum Physics,”

Day 22:
Explore any open dataset from lattice QCD or QCD-based simulation. Start a Python script to parse it.
Reference: [SciPy Lattice QCD Tools / HPC projects on GitHub]

Day 23:
Attempt a small classification or regression project with QCD data using PyTorch.
Reference: [PyTorch official tutorial + Kaggle or Zenodo QCD data if available]

Day 24:
Investigate HPC optimization strategies for large QCD arrays (use cuSPARSE or cuSOLVER if relevant).
Reference: cuSPARSE Documentation, cuSOLVER Docs

Day 25:
Try a baseline HPC pipeline for QCD data using sparse matrices on GPU.
Reference: NVIDIA HPC blog on sparse computations

---
Block 6 (Days 26–30)

Day 26:
Return to NeRF concepts: Explore how to incorporate fluid fields (like MHD data) into a NeRF pipeline (semi-experimental).
Reference: [Neural Volumes / Hybrid NeRF papers on ArXiv]

Day 27:
Basic code attempts at a “Neural Volume” approach: represent MHD 3D fields for rendering.
Reference: “Neural Volumes” by Lombardi et al.

Day 28:
Investigate quantization in neural networks: post-training quantization basics (INT8, etc.).
Reference: TensorRT 8.6 or 9.0 (Preview) INT8 Docs

Day 29:
Experiment with quantization aware training in PyTorch on a smaller CNN or MLP.
Reference: PyTorch Quantization Docs

Day 30:
Explore how to do partial quantization on a small NeRF or MLP model (just a test).
Reference: [TinyNeRF + PyTorch Quantization blog posts / repos]

---
Block 7 (Days 31–35)

Day 31:
Introduction to LLMs and QLoRA concept.
Reference: QLoRA paper (Dettmers et al.)

Day 32:
Set up a local environment (HF Transformers, bitsandbytes) for 4-bit or 8-bit LLM training.
Reference: Hugging Face PEFT + QLoRA docs

Day 33:
Fine-tune a small LLaMA-like or GPT-NeoX model on a minimal dataset with QLoRA (16GB VRAM constraint).
Reference: [QLoRA examples on Hugging Face GitHub]

Day 34:
Try an MHD or fluid small dataset in a “Tabular LLM” scenario: see if it can do basic MHD question answering.
Reference: [“Tabular LLM” – Tools or HF community posts discussing tabular data + LLM]

Day 35:
Test and record inference performance with different quantization levels (4-bit vs. 8-bit).
Reference: bitsandbytes GitHub + HF Transformers Integration Docs

---
Block 8 (Days 36–40)

Day 36:
Introduction to DocLLMs (document Q&A flows) and retrieval-augmented generation.
Reference: LangChain Docs

Day 37:
Implement a small “DocLLM” pipeline locally using a vector store (FAISS) + QLoRA LLM.
Reference: [Hugging Face Inference or LangChain tutorials]

Day 38:
Insert an MHD text resource or fluid doc into the vector store. Test question answering.
Reference: [FAISS GitHub, LangChain Examples]

Day 39:
Investigate large vision-language models (LVLMs) for fluid image or volumetric data + textual descriptions.
Reference: [BLIP, Florence, or CLIP-based LVLM approaches on Hugging Face]

Day 40:
Try a minimal CLIP-based approach to label MHD visualizations with text prompts.
Reference: OpenAI CLIP GitHub

---
Block 9 (Days 41–45)

Day 41:
GPU-based kernel programming deeper dive: advanced warp shuffles, memory coalescing, etc.
Reference: NVIDIA CUDA Best Practices Guide

Day 42:
Implement a custom reduction kernel for large arrays (sum, min, max) using warp intrinsics.
Reference: NVIDIA reduce.cu sample

Day 43:
Explore CUDA IPC to share data between processes.
Reference: CUDA IPC docs in Programming Guide

Day 44:
Attempt a multi-process pipeline: one process does PDE updates, another does some analysis. Share data via IPC.
Reference: [NVIDIA Developer Forum / HPC discussion boards]

Day 45:
Practice with NVRTC to compile CUDA kernels at runtime from strings.
Reference: NVRTC documentation

---
Block 10 (Days 46–50)

Day 46:
Dive into NVJPEG library for GPU-based image decoding.
Reference: NVJPEG Documentation

Day 47:
Integrate NVJPEG into a data-loading pipeline for training a CV model on MHD images or other fluid images.
Reference: NVIDIA DALI or NVJPEG examples on GitHub

Day 48:
Explore NPP (NVIDIA Performance Primitives) for image transformations on GPU.
Reference: NPP Docs

Day 49:
Implement a data augmentation pipeline using NPP (rotate, scale, etc.) for fluid snapshots.
Reference: [NPP sample code in the CUDA Toolkit]

Day 50:
Investigate NVGRAPH for GPU-based graph analytics; try a small GNN or GCN approach.
Reference: NVGRAPH Documentation

---
Block 11 (Days 51–55)

Day 51:
Explore GNN libraries (PyTorch Geometric or DGL) + GPU graph computations for MHD adjacency or QCD data.
Reference: PyTorch Geometric Docs

Day 52:
Implement a simple GCN for classifying a small graph-based dataset.
Reference: GCN original paper (Kipf & Welling)

Day 53:
Try advanced GNN approach for fluid mesh data or QCD lattice connectivity.
Reference: [DGL + scientific computing examples]

Day 54:
Evaluate HPC optimization for GNN training using multi-stream concurrency.
Reference: CUDA Streams doc

Day 55:
Return to TensorRT for potential GNN or CNN inference optimization.
Reference: TensorRT Docs

---
Block 12 (Days 56–60)

Day 56:
TensorRT-LLM: test or read about the new features for LLM inference acceleration.
Reference: [NVIDIA TensorRT-LLM blog posts]

Day 57:
Attempt to convert a GPT-like model to ONNX and then to TensorRT for faster inference.
Reference: [TensorRT + ONNX conversion tutorials]

Day 58:
Measure performance improvements with INT8 or FP16 on the 4090 for your fine-tuned QLoRA model.
Reference: NVIDIA Nsight Systems for profiling

Day 59:
Quick introduction to DeepSpeed for large model training on limited GPU memory.
Reference: DeepSpeed GitHub & Tutorials

Day 60:
Attempt partial pipeline parallel or ZeRO optimization with DeepSpeed on a small GPT or BERT.
Reference: DeepSpeed ZeRO docs

---

Block 13 (Days 61–65)

Day 61:
Tackle DirectX on Windows: basics of the GPU pipeline, creating a device/context.
Reference: DirectX 12 Intro Docs

Day 62:
Render a simple triangle in DirectX 12.
Reference: DirectX 12 samples from Microsoft

Day 63:
Investigate compute shaders in DirectX for GPGPU tasks.
Reference: DirectCompute documentation

Day 64:
Attempt a small fluid simulation or parted PDE approach with DirectCompute.
Reference: [GPU Pro books / Microsoft samples]

Day 65:
Compare DirectCompute performance to your CUDA approach.
Reference: [Forums / Articles comparing DirectCompute vs CUDA]

---
Block 14 (Days 66–70)

Day 66:
OpenGL basics: create a simple context in WSL2 using X11 forwarding or Windows host.
Reference: LearnOpenGL

Day 67:
Draw a rotating 3D shape with modern OpenGL (VAOs, VBOs, shaders).
Reference: [OpenGL Superbible or any up-to-date tutorial]

Day 68:
Explore OpenGL ES differences. Possibly use an emulator or angle to see it in action.
Reference: OpenGL ES docs

Day 69:
Start linking an OpenGL “surface visualization” to your MHD data (render isosurfaces).
Reference: [Marching Cubes GPU implementations, e.g., NVIDIA sample]

Day 70:
Basic overlay of CUDA–OpenGL interop to display fluid fields in real time.
Reference: [CUDA-OpenGL interop sample in the CUDA toolkit]

---
Block 15 (Days 71–75)

Day 71:
Check out Vulkan basics, especially compute pipelines.
Reference: Vulkan Tutorial

Day 72:
Implement a minimal Vulkan compute pipeline for a simple operation (vector add).
Reference: Official Khronos Vulkan Samples

Day 73:
Integrate Vulkan + MHD data to do some GPU-based fluid step.
Reference: [Vulkan compute examples on GitHub]

Day 74:
Investigate X11 usage under WSL2, how to run Vulkan or OpenGL with X forwarding.
Reference: [X11 in WSL2 docs or community guides]

Day 75:
Learn about OpenVG, EGL, EGLOutput, and EGLSync for low-level rendering on embedded devices.
Reference: Khronos EGL docs

----
Block 16 (Days 76–80)

Day 76:
Build a minimal EGL + OpenGL ES environment on WSL2 if possible.
Reference: [EGL/GL ES on Linux tutorials]

Day 77:
Explore NVSCI and NvMedia (NVIDIA APIs for advanced cross-process video/graphics).
Reference: [NVIDIA Drive or Jetson documentation (NvMedia / NVSCIIPC)]

Day 78:
Attempt a minimal pipeline combining an EGL surface + CUDA to display real-time fluid updates.
Reference: [NVIDIA Drive/Jetson sample code for display integration]

Day 79:
Expand to handle multiple buffers with EGLSync for synchronization.
Reference: [Khronos EGLSync docs]

Day 80:
Evaluate performance for real-time MHD or fluid data visualization pipeline.
Reference: [Nsight Graphics or Nsight Systems for profiling]

---
Block 17 (Days 81–85)

Day 81:
Deeper look at Quantum Machine Learning with PennyLane’s advanced demos.
Reference: PennyLane QML demos

Day 82:
Attempt a hybrid classical–quantum model for a small dataset.
Reference: [PennyLane tutorial on hybrid models]

Day 83:
Investigate parametric circuits for classification or regression tasks, perhaps fluid classification.
Reference: [Qiskit Machine Learning’s QSVM or VQC demos]

Day 84:
Run or emulate a small quantum device approach to a fluid PDE (toy model).
Reference: [Quantum PDE solving references on arXiv]

Day 85:
Start reading about HPC + quantum co-processing or quantum-inspired algorithms.
Reference: [Quantum-Inspired HPC docs from D-Wave or IonQ blog]

---
Block 18 (Days 86–90)

Day 86:
Basic introduction to Quantum Chromodynamics-based ML more thoroughly. Possibly check PyTorch for implementing small lattice updates.
Reference: [Research papers on lattice QCD ML approaches, e.g., “ML for LQCD” on arXiv]

Day 87:
Attempt a small HPC code to handle mini-lattice updates with CUDA.
Reference: [GitHub repos for Lattice QCD on GPUs (QUDA library if open-sourced)]

Day 88:
Evaluate hooking in QML ideas to your HPC-lattice code (this is experimental).
Reference: [QUDA (Quantum Chromodynamics on GPUs) library docs if you can find it]

Day 89:
Investigate Tensor Cores for HPC matrix ops.
Reference: [NVIDIA Ampere or Ada GPU Tensor Core docs]

Day 90:
Attempt to use Tensor Cores in a custom kernel (WMMA) or with cuBLAS “tensor op” calls.
Reference: [CUDA WMMA official sample code]

---
Block 19 (Days 91–95)

Day 91:
Return to NeRF: check advanced repos (NeRFStudio, Instant-NGP).
Reference: Instant-NGP by NVIDIA

Day 92:
Install and run instant-ngp on your MHD volume or fluid data (converted to “images” or partial data).
Reference: [Nerfstudio data loading docs]

Day 93:
Investigate potential real-time rendering or partial 3D reconstruction for fluid “volume rendering.”
Reference: [NeRF for volume rendering - specialized research papers]

Day 94:
Start linking the concept of “Neural Radiance Fields in Fluids” more concretely.
Reference: [Research on “Neural Fields for PDE or fluid simulation” on ArXiv]

Day 95:
Experiment with training a small MHD “Neural Field” to predict next time-step.
Reference: [Neural PDE approximation papers (Fourier Neural Operator, etc.)]

---
Block 20 (Days 96–100)

Day 96:
Dive deeper into quantization for NeRF: try post-training quantization on an Instant-NGP model.
Reference: [TensorRT INT8 for NeRF discussion on forums or GitHub forks]

Day 97:
Evaluate speed vs. accuracy trade-offs with INT8 NeRF.
Reference: [Nsight Systems to measure performance]

Day 98:
Explore Quantization Aware Training for a small neural volume model.
Reference: [PyTorch QAT docs again]

Day 99:
Check if you can combine QLoRA ideas with a smaller neural field for memory savings.
Reference: [LoRA in non-transformer contexts, community experiments]

Day 100:
Possibly combine all: HPC fluid PDE → Neural Field representation → Quantized inference.
Reference: [Your own HPC code integrated with small MLP for PDE steps]

---
Block 21 (Days 101–105)

Day 101:
CUDA graph features for capturing and replaying GPU workloads.
Reference: CUDA Graphs doc

Day 102:
Convert your PDE or MHD pipeline into a CUDA graph to see performance gain.
Reference: [CUDA Graphs sample in the toolkit]

Day 103:
Investigate using multiple GPUs if you ever add an external GPU or HPC cluster.
Reference: NCCL docs (NVIDIA Collective Communications Library)

Day 104:
Explore concurrency with streams, events, pinned memory for faster transfers.
Reference: CUDA Streams and Events Guide

Day 105:
Try splitting your PDE steps among multiple streams or concurrency to see speedups.
Reference: [Multi-stream concurrency sample codes]

----
Block 22 (Days 106–110)

Day 106:
Revisit GNN pipelines with large graphs or adjacency from fluid grids.
Reference: [PyTorch Geometric advanced features]

Day 107:
Use NVGraph or cuSPARSE for better graph ops.
Reference: cuGraph from RAPIDS

Day 108:
Benchmark GNN training vs. standard CPU approach.
Reference: [cugraph docs + PyTorch Geometric + RAPIDS integration]

Day 109:
Attempt advanced GCN or Graph Attention Networks (GAT) on HPC data.
Reference: GAT paper (Velickovic et al.)

Day 110:
Integrate your concurrency knowledge to speed up dataset prefetch / augmentation.
Reference: [NVIDIA DALI or custom concurrency pipelines]

---
Block 23 (Days 111–115)

Day 111:
Return to Quantum side: try a small “Variational Quantum Eigensolver” (VQE) for a simple chemistry problem.
Reference: [Qiskit Chemistry / PennyLane Chemistry demos]

Day 112:
Evaluate HPC-lattice synergy with quantum circuits: advanced experimental area.
Reference: [Papers on quantum-lattice synergy]

Day 113:
Investigate quantum error mitigation or noise models in simulators.
Reference: [Qiskit Ignis or Mitiq library]

Day 114:
Attempt noise-aware QML model on a simulator.
Reference: [Mitiq docs or Qiskit Aer]

Day 115:
Try scaling your QML model if possible with GPU-based simulators.
Reference: [NVIDIA cuQuantum library or Qiskit Aer GPU docs]

---
Block 24 (Days 116–120)

Day 116:
Coding-based LLMs: check out CodeLLMs (StarCoder, CodeLlama) for GPU usage.
Reference: StarCoder Hugging Face model card

Day 117:
Attempt local fine-tuning with QLoRA for code generation tasks.
Reference: [PEFT/QLoRA for StarCoder examples on HF]

Day 118:
Evaluate 4-bit vs. 8-bit for code LLM, test on your 16GB VRAM.
Reference: [bitsandbytes integration for code LLMs]

Day 119:
See if you can get the LLM to help generate your HPC kernels or PDE code.
Reference: [LangChain “agent” approach for code generation]

Day 120:
Explore performance or accuracy metrics for coding tasks with small local LLM.
Reference: [HumanEval or MBPP dataset for code generation benchmarks]

---
Block 25 (Days 121–125)

Day 121:
Explore DocLLMs specifically for HPC or CUDA documentation search.
Reference: [LangChain docs for doc-based QA, integrating CUDA docs offline]

Day 122:
Index the official CUDA docs, HPC references, and see if your local LLM can answer HPC questions.
Reference: [FAISS / Chroma / Milvus vector DB for doc chunking]

Day 123:
Add a small “chatbot” interface to ask HPC / quantum queries.
Reference: [Gradio or Streamlit-based chat UI]

Day 124:
Test out advanced retrieval-based question answering, ensuring correct context retrieval.
Reference: [LangChain RetrievalQA or RAG approach]

Day 125:
Evaluate the possibility of fine-tuning for HPC domain knowledge.
Reference: [PEFT + domain text approach]

---
Block 26 (Days 126–130)

Day 126:
Investigate DirectX Raytracing (DXR) to visualize volumetric data.
Reference: Microsoft DXR samples on GitHub

Day 127:
Attempt a minimal ray tracing example in DirectX 12.
Reference: [NV Raytracing tutorials if any for Windows]

Day 128:
Combine DXR with your fluid volume for advanced real-time visualization.
Reference: [Raymarched volume examples in DirectX or Vulkan]

Day 129:
Compare DXR vs. path tracing in OpenGL or Vulkan.
Reference: Ray tracing in Vulkan (VK_KHR_ray_tracing)

Day 130:
See if real-time volumetric rendering can integrate with your HPC PDE updates.
Reference: [Nsight Graphics for debug]

---
Block 27 (Days 131–135)

Day 131:
Return to NPP or your image pipeline: advanced morphological ops, filtering, etc.
Reference: [NPP advanced filtering sample code]

Day 132:
Integrate morphological transformations for segmentation tasks in fluid frames.
Reference: [Any morphological operation doc (OpenCV + GPU or NPP docs)]

Day 133:
Investigate advanced custom GPU kernels for wavelet transforms in fluid analysis.
Reference: [Wavelet transform HPC examples or cuFFT + wavelet combos]

Day 134:
Attempt a wavelet-based decomposition of an MHD field and visualize results.
Reference: [PyWavelets, custom GPU wavelet code on GitHub]

Day 135:
If you see interesting patterns, feed them into your LLM as “image + text” for summarization.
Reference: [CLIP or BLIP for bridging image <-> text]

---
Block 28 (Days 136–140)

Day 136:
Investigate CUSOLVER for advanced linear algebra on GPU (eigenvalue decomposition, SVD).
Reference: CUSOLVER docs

Day 137:
Explore how to handle large matrix operations for PDE or QCD systems with CUSOLVER.
Reference: [Examples in cuSOLVER GitHub or HPC tutorials]

Day 138:
Implement a GPU-based SVD for some MHD or QCD data snapshot matrix.
Reference: [NVIDIA HPC code samples or custom code]

Day 139:
Investigate using SVD-based dimension reduction for fluid data and feed into your GNN or LLM.
Reference: [PCA/SVD dimension reduction HPC references]

Day 140:
Evaluate performance vs. CPU-based SVD.
Reference: [Nsight Systems / Nsight Compute for performance measurement]

----
Block 29 (Days 141–145)

Day 141:
Quantum-inspired algorithms: glimpses into the “Trove of Quantum-inspired HPC.”
Reference: [Microsoft’s Quantum-Inspired Optimization docs]

Day 142:
Attempt a small quantum-inspired approach for fluid optimization or PDE solver.
Reference: [Papers on quantum annealing for PDE or lattice QCD]

Day 143:
If feasible, adapt a QAOA-based approach for a discrete optimization in your HPC code.
Reference: [Qiskit or PennyLane QAOA tutorials]

Day 144:
Compare results with classical HPC solutions.
Reference: [Mitiq or Qiskit measurement tools]

Day 145:
Evaluate potential future directions (scalable HPC or real quantum hardware usage).
Reference: [IonQ, Rigetti, IBM Q for possible future runs]

---
Block 30 (Days 146–150)

Day 146:
Revisit CUDA kernel optimization: instruction-level optimization, occupancy analysis.
Reference: [NVIDIA Occupancy Calculator in Nsight Compute]

Day 147:
Fine-tune your PDE or MHD kernel to reduce registers, improve occupancy.
Reference: [NVIDIA HPC webinar or dev blog on occupancy]

Day 148:
Explore CUDA Inline PTX for advanced control.
Reference: [Inline PTX docs in CUDA C++ Programming Guide]

Day 149:
Attempt a small segment of your solver with inline PTX for specialized instructions.
Reference: [Examples on GitHub / HPC dev blogs]

Day 150:
Evaluate performance improvements.
Reference: [Nsight Compute for kernel analysis]

----
Block 31 (Days 151–155)

Day 151:
CURAND deeper usage: advanced random distributions for fluid or quantum simulations.
Reference: [CURAND docs advanced usage]

Day 152:
Integrate random fluctuations in MHD or QCD code to represent noise or initial conditions.
Reference: [Physical modeling references for random fields]

Day 153:
Investigate parallel random number generation performance.
Reference: [NVIDIA dev blog on parallel RNG]

Day 154:
Explore using sobol or quasi-random sequences for HPC or QML.
Reference: [CURAND docs on sobol sequences]

Day 155:
Try a quick experiment with variance reduction in PDE or quantum circuit simulation.
Reference: [Papers on random sampling methods in HPC]

----
Block 32 (Days 156–160)

Day 156:
Shift attention to advanced DeepSpeed pipeline or model parallel: train bigger LLM on HPC data.
Reference: [DeepSpeed pipeline parallel docs]

Day 157:
Evaluate memory usage with ZeRO stage 2 or 3 on your 16GB GPU.
Reference: [DeepSpeed advanced tutorials]

Day 158:
Attempt multi-node simulation on a cloud instance if possible (optional).
Reference: [Azure or AWS HPC resources + DeepSpeed docs]

Day 159:
Integrate QLoRA + DeepSpeed for large code or HPC domain LLM.
Reference: [Any community examples combining QLoRA + DS]

Day 160:
Validate results with HPC Q&A or PDE code generation tasks.
Reference: [HF Transformers + DS + local dataset pipeline]

Block 33 (Days 161–165)

Day 161:
Explore advanced NeRF variants (e.g., Ref-NeRF, TensoRF) for better efficiency.
Reference: TensoRF paper

Day 162:
Check if TensoRF can compress 3D volumetric data for fluid or MHD.
Reference: [TensoRF GitHub repos or “compressing neural fields” research]

Day 163:
Attempt training TensoRF on a toy MHD dataset.
Reference: [Implementation instructions from the TensoRF repo]

Day 164:
Evaluate GPU memory usage and see if quantization helps.
Reference: [Any INT8 or half-precision TensoRF forks]

Day 165:
Compare reconstruction error to your earlier NeRF approach.
Reference: [Custom scripts or HPC analysis]

---
Block 34 (Days 166–170)

Day 166:
OpenVG specifics on embedded GPUs if you plan to deploy fluid viz on a smaller device.
Reference: [Khronos OpenVG docs]

Day 167:
Explore vector graphics acceleration to see if relevant to MHD overlays or HPC dashboards.
Reference: [NVIDIA Jetson / Drive docs for advanced GPU usage]

Day 168:
Basic project: use OpenVG to render a set of vector shapes from fluid data.
Reference: [OpenVG sample code]

Day 169:
Attempt an EGL + OpenVG + CUDA pipeline (very niche).
Reference: [Some embedded system GPU pipeline references]
Day 170:

Compare performance to an OpenGL-based overlay.
Reference: [Nsight Systems for performance counters]
Block 35 (Days 171–175)
Day 171:

Go deeper into NVSCI: advanced inter-process and cross-domain buffering.
Reference: [NVIDIA documentation for NVSCI on Jetson/Drive platforms]
Day 172:

Attempt a minimal pipeline that uses NVSCI buffers to share data between a CUDA process and a rendering process.
Reference: [Example code if available from NVIDIA dev forums]
Day 173:

Explore NvMedia for capturing or encoding fluid animation frames.
Reference: [NvMedia docs for image & video processing on GPU]
Day 174:

Implement a pipeline: fluid simulation → shared memory via NVSCI → compressed via NvMedia → display.
Reference: [Integration guides for NVSCI + NvMedia on dev forums]
Day 175:

Evaluate if this approach is feasible on your Windows WSL2 environment or purely hypothetical.
Reference: [NVIDIA forum discussions about NVSCI support on Windows]
Block 36 (Days 176–180)
Day 176:

Examine TensorRT-LLM updates for the newest large models.
Reference: [NVIDIA developer blogs or official GitHub repos]
Day 177:

Try building a custom engine for a popular model like Llama2 or CodeLlama.
Reference: [TensorRT build steps for Llama2 if available]
Day 178:

Evaluate real-time inference speeds for HPC question answering.
Reference: [Hugging Face “transformers-cli benchmark” or TRT profiling]
Day 179:

Investigate multi-context parallelism in TensorRT for consecutive LLM requests.
Reference: [TensorRT multi-context docs]
Day 180:

If time, test an end-to-end HPC doc chatbot with TensorRT-accelerated LLM.
Reference: [LangChain or custom pipeline integration tutorial]
Block 37 (Days 181–185)
Day 181:

CUBLAS advanced features: batched operations for many small matrices.
Reference: [cublas<t>batched function docs]
Day 182:

Implement batched PDE solver steps that rely on small matrix operations.
Reference: [NVIDIA HPC PDE code samples or your own PDE approach]
Day 183:

Investigate using tensor cores for half-precision matmul.
Reference: [Mixed-precision training in HPC contexts, e.g. HPC slides from NVIDIA]
Day 184:

Compare single precision vs. half precision for PDE accuracy vs. speed.
Reference: [Nsight for profiling again]
Day 185:

Possibly integrate that into your real-time fluid or MHD pipeline.
Reference: [Your existing PDE code]
Block 38 (Days 186–190)
Day 186:

Revisit quantum circuits with a focus on QCD or HPC-like operators.
Reference: [Advanced QCD references from arXiv + quantum computing papers]
Day 187:

Attempt to build a small HPC–Quantum synergy example: a PDE step on GPU, partially guided by quantum circuit.
Reference: [Quantum Approximate Optimization Algorithm references]
Day 188:

Investigate parallelizing your quantum simulator across multiple CPU threads or GPU.
Reference: [Qiskit Aer GPU mode, cuQuantum usage]
Day 189:

Check performance vs. a purely classical HPC approach.
Reference: [Nsight CPU + GPU concurrency analysis]
Day 190:

Keep the best working synergy approach for future expansions.
Reference: [Your HPC or QML codebase]
Block 39 (Days 191–195)
Day 191:

Look into specialized HPC frameworks like OpenACC or OpenMP for GPU.
Reference: [OpenACC.org docs]
Day 192:

Attempt porting a PDE loop from CUDA C to OpenACC.
Reference: [PGI/NVIDIA HPC SDK docs on OpenACC]
Day 193:

Compare performance and coding complexity between pure CUDA vs. OpenACC.
Reference: [NVIDIA HPC SDK performance guides]
Day 194:

Explore if you can combine GPU-accelerated OpenACC code with your LLM or QML approach.
Reference: [Mixed HPC approaches discussion in HPC forums]
Day 195:

Evaluate HPC technique readiness for quantum HPC or advanced fluid codes.
Reference: [Research HPC frameworks + future directions]
Block 40 (Days 196–200)
Day 196:

Revisit Quantum Machine Learning for classification tasks on HPC data.
Reference: [PennyLane or Qiskit ML advanced classification demos]
Day 197:

Attempt a small specialized model (e.g., anomaly detection in MHD snapshots) with a quantum classifier.
Reference: [Quantum anomaly detection papers]
Day 198:

Optimize data pre-processing on GPU, then feed to quantum circuit.
Reference: [NVIDIA HPC or RAPIDS for data cleaning, then Qiskit interface]
Day 199:

Evaluate classification accuracy vs. classical HPC approach.
Reference: [Comparison methodology from QML papers]
Day 200:

Combine code LLM to automatically generate quantum circuit scaffolding.
Reference: [LangChain agents for code generation + Qiskit]
Block 41 (Days 201–205)
Day 201:

CURAND advanced distribution usage for debugging or stress testing MHD PDE.
Reference: [CURAND docs again, focus on distribution reliability]
Day 202:

Try seeding fluid PDE with random initial conditions in a large parameter space.
Reference: [Monte Carlo style HPC PDE solutions]
Day 203:

Evaluate if code LLM can help auto-tune PDE solver parameters.
Reference: [OpenAI or local code LLM prompts to tune HPC code]
Day 204:

Try generating many solutions in parallel, capturing results in an HPC pipeline.
Reference: [Multi-stream concurrency or HPC job scheduling docs]
Day 205:

Possibly feed the PDE results back into a training pipeline for a neural PDE approach.
Reference: [Deep Operator Networks or FNO references]
Block 42 (Days 206–210)
Day 206:

Investigate NVRTC again for JIT compilation of PDE kernels from user input (like a code LLM output).
Reference: [NVRTC docs]
Day 207:

Auto-generate a PDE kernel with LLM, compile via NVRTC, run it in real time.
Reference: [Integration approach with Python + NVRTC C++ calls]
Day 208:

Test correctness with simple PDE (heat equation, wave equation).
Reference: [Comparative HPC PDE references]
Day 209:

Expand to more complex fluid PDE or partial MHD.
Reference: [Your MHD solver code + new kernel generation approach]
Day 210:

Evaluate speed and correctness vs. manually coded approach.
Reference: [Nsight or HPC verification approach]
Block 43 (Days 211–215)
Day 211:

Investigate advanced quantization for large LLMs: ZeroQuant, etc.
Reference: [ZeroQuant MSR papers on 8-bit or 4-bit quant]
Day 212:

Attempt post-training quantization on a bigger GPT model you have.
Reference: [ONNX Runtime quantization or PyTorch approach]
Day 213:

Compare accuracy to QLoRA approach.
Reference: [Benchmarks from Hugging Face community]
Day 214:

Check synergy of quantization + QLoRA + HPC domain fine-tuning.
Reference: [Any research or GitHub repos combining these]
Day 215:

Evaluate inference speed gains on the 4090.
Reference: [Nsight for final measurements]
Block 44 (Days 216–220)
Day 216:

Neural Radiance Fields in Fluids deeper approach: 4D (3D + time) representation.
Reference: [Dynamic NeRF or “D-NeRF” papers]
Day 217:

Attempt coding or adapting a D-NeRF approach for your PDE time-series data.
Reference: [D-NeRF official GitHub if it exists]
Day 218:

Evaluate GPU memory usage and see if half precision or QAT helps.
Reference: [PyTorch half-precision training tips]
Day 219:

Investigate extracting physically meaningful flows from the learned representation.
Reference: [Physical interpretability of neural fields in fluid domain papers]
Day 220:

Compare performance to a standard HPC PDE solver.
Reference: [Your HPC PDE code + D-NeRF results]
Block 45 (Days 221–225)
Day 221:

Deeper DirectX12 compute pipeline usage for large data.
Reference: [MS docs on resource binding, descriptor heaps, etc.]
Day 222:

Attempt MHD PDE steps in a DirectX12 compute pipeline for curiosity.
Reference: [DirectCompute or DX12 compute tutorial]
Day 223:

Compare kernel development difficulty vs. CUDA.
Reference: [Your existing PDE code in CUDA]
Day 224:

Investigate synergy with ray-tracing pipelines for fluid visualization in DX12.
Reference: [DXR docs again + HPC fluid demos if any]
Day 225:

Evaluate how to share data between DX12 and CUDA with interop.
Reference: [NVIDIA docs on DX–CUDA interop]
Block 46 (Days 226–230)
Day 226:

Consider advanced HPC load balancing with multi-GPU or multi-CPU scheduling (slurm-like on single machine).
Reference: [HTCondor, Slurm, or local Windows HPC scheduling approaches]
Day 227:

Attempt concurrency: PDE solver on GPU + QLoRA fine-tuning in background (just a stress test).
Reference: [Multi-process GPU concurrency docs]
Day 228:

Investigate using Docker or Singularity for containerizing HPC + LLM pipeline.
Reference: [NVIDIA Container Toolkit docs]
Day 229:

Test overhead of containerization on WSL2.
Reference: [Docker + WSL2 docs]
Day 230:

Keep refining HPC + LLM pipeline within container for consistent environment.
Reference: [Multi-stage Dockerfiles with CUDA support]
Block 47 (Days 231–235)
Day 231:

Expand GNN usage: try a spatio-temporal GNN for fluid or MHD.
Reference: [ST-GCN or temporal GNN papers]
Day 232:

Build dataset from PDE time steps as graph sequences.
Reference: [PyTorch Geometric temporal submodule]
Day 233:

Train ST-GCN on GPU, see if it can predict next state.
Reference: [Temporal GNN examples in PyTorch Geometric]
Day 234:

Compare performance to your earlier NeRF-based or neural PDE approaches.
Reference: [Your own code benchmarks]
Day 235:

If feasible, combine partial HPC PDE + GNN correction step.
Reference: [Hybrid HPC-ML PDE research]
Block 48 (Days 236–240)
Day 236:

Look into NVGraph again for large dynamic graphs.
Reference: [NVGraph docs, cugraph for dynamic graphs?]
Day 237:

Evaluate partial real-time graph partitioning of fluid cells or QCD lattice.
Reference: [Partitioning HPC references, cugraph calls]
Day 238:

Integrate GNN training in real-time with the partitioning approach.
Reference: [cugraph + PyTorch Geometric or DGL synergy]
Day 239:

Explore advanced kernel fusion to reduce memory passes.
Reference: [NVIDIA HPC blog on kernel fusion or custom approach]
Day 240:

Profile everything, aim for sub-second updates on your data.
Reference: [Nsight Systems thorough analysis]
Block 49 (Days 241–245)
Day 241:

Investigate advanced Quantum computing hardware (IonQ, Rigetti, IBM). Possibly sign up for free demos.
Reference: [IBM Quantum or IonQ documentation]
Day 242:

Try running your QML or quantum PDE code on real hardware if you have credits.
Reference: [Qiskit or IonQ tutorial with real device usage]
Day 243:

Compare the difference in results vs. simulation.
Reference: [Qiskit’s backend properties / error rates]
Day 244:

Evaluate if quantum hardware helps or is too noisy for HPC PDE tasks.
Reference: [Your results, error mitigation approaches]
Day 245:

Potential next expansions: big HPC cluster usage or specialized quantum HPC.
Reference: [Frontier HPC or HPC+Quantum synergy articles]
Block 50 (Days 246–250)
Day 246:

Explore TensorRT-LLM or DeepSpeed + QLoRA on the largest possible model your 16GB VRAM can handle.
Reference: [HF Transformers to see memory usage of bigger LLMs, e.g., 13B or 30B param at 4-bit]
Day 247:

Fine-tune that model on your HPC/Quantum-coded knowledge base.
Reference: [PEFT approach with domain text data]
Day 248:

Evaluate inference speed, compare to smaller models you tested previously.
Reference: [Nsight or HF performance measurement]
Day 249:

Attempt an integrated final pipeline: HPC PDE solver + real-time fluid visualization + LLM Q&A about MHD or QCD.
Reference: [Your combined code from prior blocks]
Day 250:

Explore advanced HPC synergy: dynamic parallel PDE, GNN or NeRF-based fluid representation, QLoRA LLM for code suggestions, maybe quantum circuit for certain sub-problems. Keep coding deeper every day.
Reference: [All your previous scripts, HPC + ML + Quantum docs from earlier blocks]
