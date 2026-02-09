# CUDARFI

Acoustic Radiation Force Impulse imaging is a non-invasive ultrasound technique for measuring tissue deformation.
This is a high-performance delay-and-sum beamformer implementation using CUDA for GPU acceleration.
It can be used to process RF channel data into B-mode images and videos.

You can find the original example [here](http://www.ustb.no/examples/acoustical-radiation-force-imaging/arfi-from-uff-file-recorded-with-verasonics/).

![B-Mode Image Output](b_mode.png)
![Displacement Movie](wave.mp4)

## Dependencies

- **CUDA Toolkit** (11.0+)
- **HDF5** (C++ bindings)
- **FFTW3** (Float precision `fftw3f`)
- **OpenMP**
- **CMake** (3.10+)
- **Make**

### Python Visualization

The included `visualize.py` script requires a few additional Python packages:

- **Python 3**
- **h5py**: For reading the HDF5 output.
- **NumPy**: For numerical array manipulation.
- **Matplotlib**: For plotting images.
- **SciPy**: For the Kasai autocorrelation algorithm (specifically `ndimage`).
- **FFmpeg**: For generating the output video (`wave.mp4`).

## Installation & Build

The project includes a `Makefile` for easy building.
If you have `nix` installed, you can use `NIXPKGS_ALLOW_UNFREE=1 nix-shell .` to install everything that is required.

> **Note:** I have not tested if this runs using the CUDA package nix provides because I only have an amd gpu locally.

```bash
# Build the project
make build

# Clean build artifacts
make clean
```

## Usage

The code is written to use the ARFI dataset from USTB. It's expected to be in the project root directory as `ARFI_dataset.uff` and can be downloaded [here](http://www.ustb.no/datasets/ARFI_dataset.uff).

You can run different versions of the beamformer using `make` targets:

```bash
# Run CPU version (Multi-core OpenMP)
make run

# Run Naive CUDA version
make run-cuda

# Run Optimized CUDA version
make run-cuda-opt

# View all available commands
make help
```

## Performance Comparison

Benchmarks run on `ARFI_dataset.uff`.

| Implementation | Execution Time | Speedup vs CPU | Hardware |
|:---|:---:|:---:|:---|
| **CPU (OpenMP)** | 1.4375 s | 1x | 7800X3D & DDR5-6000 |
| **CUDA Naive** | 0.8025 s | 1.79× | RTX 3060 |
| **CUDA Optimized** | **0.6500 s** | **2.21×** | RTX 3060 |

> **Note:** The beamforming algorithm is fundamentally **memory-bound**.

## Algorithm

The core principle is to coherent sum the signals recorded by each transducer element, applying the correct time delays to focus the beam at every pixel in the image.

For a pixel at location $(x, z)$, the total time of flight $t(x, z, i)$ for the $i$-th element is calculated as:

$$
t(x, z, i) = t_{tx}(z) + t_{rx}(x, z, i)
$$

Where:

**1. Transmit Delay ($t_{tx}$)**

Assuming a plane wave transmission, the time for the wave to reach depth $z$ is:

$$
t_{tx}(z) = \frac{z}{c} + t_0
$$

($c$ is the speed of sound, $t_0$ is the start time)

**2. Receive Delay ($t_{rx}$)**

The time for the echo to return from the pixel to the $i$-th transducer element at position $(x_i, z_i)$ is the geometric distance divided by the speed of sound:

$$
t_{rx}(x, z, i) = \frac{\sqrt{(x - x_i)^2 + (z - z_i)^2}}{c}
$$

The final beamformed value $I(x, z)$ is obtained by summing the interpolated signal samples $S_i(t)$ from all $N$ channels:

$$
I(x, z) = \sum_{i=1}^{N} S_i \left( (t(x, z, i) - t_0) \cdot f_s \right)
$$

($f_s$ is the sampling frequency)

Linear interpolation is used to fetch samples at sub-sample precision, reducing artifacts and improving image quality.

## CUDA Optimizations

### 1. Texture Cache
**Why:** Beamforming involves a "scatter-gather" memory access pattern where adjacent pixels might need data from widely separated memory locations (due to time delays). This defeats standard coalescing.  

**Implementation:** I used the `__ldg()` intrinsic to force loads through the read-only texture cache, which is optimized for 2D spatial locality. This provided the biggest performance jump (~10%).

### 2. Launch Bounds & Occupancy
**Why:** Without launch bounds the compiler might cause threads to use too many registers, limiting how many warps can run simultaneously (occupancy).  

**Implementation:** With `__launch_bounds__(256)` I can give the compiler the information that no more than 256 threads will be launched in a block, but those launched really need to run, thus it can optimize the register usage per thread.

### 3. Constant Memory
**Why:** Probe geometry (`el_x`, `el_z`) is constant across all pixel calculations.  

**Implementation:** Stored in `__constant__` memory, allowing broadcast reads to all threads in a warp simultaneously, rather than repeated global memory fetches.

### 4. Fast Math Intrinsics
**Implementation:**
- `__frcp_rn()`: Fast reciprocal to avoid slow division.
- `__fsqrt_rn()`: Hardware-accelerated square root.
- `__fmaf_rn()`: Fused Multiply-Add for linear interpolation.
- `__float2int_rd()`: Fast float-to-int conversion.

### 5. Precomputed Inverses
**Implementation:** Precomputed `1.0f / c` (speed of sound) on the host. Using this with multiplication instead of a division provides a small performance boost.

## Failed / Less Effective Optimizations

### 1. Shared Memory for RF Data
The specific access pattern of delay-and-sum beamforming is unpredictable and spans a large dynamic range of memory addresses. Loading a "tile" of RF data into shared memory is difficult because different threads need vastly different chunks of data depending on the steering angle and depth.

### 2. Warp Shuffles
Warp shuffle operations are great for reductions (summing values within a warp). However, each thread in our kernel computes an *independent* pixel. Because there is no need for threads to communicate or sum values with each other I figured this will not provide value.

### 3. Loop Unrolling (Compiler vs Manual)
I manually unrolled the channel loop (`#pragma unroll 8`). This provided a minor speedup by increasing instruction-level parallelism, but the gain was small compared to the memory bandwidth bottlenecks.
