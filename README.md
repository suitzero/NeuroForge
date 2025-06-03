# NeuroForge

NeuroForge is a high-performance compiler designed to optimize and map machine learning workloads onto DPU and NPU hardware. It translates complex neural operations into low-level instructions tailored for maximum efficiency, enabling fast, power-aware inference on edge and embedded systems. Whether you're building next-gen AR/VR, robotics, or intelligent networking applications, NeuroForge bridges the gap between high-level ML frameworks and specialized silicon.

## Core Components

NeuroForge is comprised of the following core components:

*   **Parser:** Responsible for reading the input neural network model (e.g., ONNX, TensorFlow Lite) and translating it into an internal representation (IR) that the compiler can understand.
*   **Optimizer:** This component takes the IR and applies a series of hardware-agnostic and hardware-specific optimizations. This includes operations like layer fusion, operator scheduling, memory layout optimization, and quantization to improve the model's performance and reduce its footprint.
*   **Hardware Abstraction Layer (HAL):** The HAL provides a standardized interface for the compiler to interact with various DPU (Data Processing Unit) and NPU (Neural Processing Unit) backends. It abstracts away the specific details of each hardware target.
*   **Code Generator:** This final component takes the optimized IR and, using the HAL, generates low-level, hardware-specific instructions or executables that can be run on the target DPU/NPU.
