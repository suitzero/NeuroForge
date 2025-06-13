# NeuroForge

NeuroForge is a high-performance compiler designed to optimize and map machine learning workloads onto DPU and NPU hardware. It translates complex neural operations into low-level instructions tailored for maximum efficiency, enabling fast, power-aware inference on edge and embedded systems. Whether you're building next-gen AR/VR, robotics, or intelligent networking applications, NeuroForge bridges the gap between high-level ML frameworks and specialized silicon.

## Core Components

NeuroForge is comprised of the following core components:

*   **Parser:** Responsible for reading the input neural network model (e.g., ONNX, TensorFlow Lite) and translating it into an internal representation (IR) that the compiler can understand.
*   **Optimizer:** This component takes the IR and applies a series of hardware-agnostic and hardware-specific optimizations. This includes operations like layer fusion, operator scheduling, memory layout optimization, and quantization to improve the model's performance and reduce its footprint.
*   **Hardware Abstraction Layer (HAL):** The HAL provides a standardized interface for the compiler to interact with various DPU (Data Processing Unit) and NPU (Neural Processing Unit) backends. It abstracts away the specific details of each hardware target.
*   **Code Generator:** This final component takes the optimized IR and, using the HAL, generates low-level, hardware-specific instructions or executables that can be run on the target DPU/NPU.

## Internal Representation (IR)

The NeuroForge Internal Representation is a graph-based data structure designed to capture the computational graph of a neural network model. It allows for various transformations and optimizations before code generation. The main components of the IR are:

*   **`Graph`**:
    *   The main container for the entire model representation.
    *   Attributes:
        *   `name`: Name of the model.
        *   `inputs`: A list of `TensorNode`s representing the primary inputs to the model.
        *   `outputs`: A list of `TensorNode`s representing the primary outputs of the model.
        *   `nodes`: A list of all `OperatorNode`s in the graph, typically in a topologically sorted order.

*   **`TensorNode`**:
    *   Represents a data tensor within the graph. This can be a model input, an operator's output, a constant weight, or a bias.
    *   Attributes:
        *   `id`: A unique identifier for this tensor node within the graph.
        *   `name`: A human-readable name for the tensor (e.g., "input_image", "conv1_weights").
        *   `shape`: A tuple or list representing the dimensions of the tensor (e.g., `(1, 3, 224, 224)`).
        *   `dtype`: The data type of the tensor's elements (e.g., "float32", "int8").
        *   `producer`: (Optional) The `OperatorNode` that produces this tensor.
        *   `consumers`: (Optional) A list of `OperatorNode`s that consume this tensor.

*   **`OperatorNode`**:
    *   Represents a computational operation in the model (e.g., convolution, ReLU, matrix multiplication).
    *   Attributes:
        *   `id`: A unique identifier for this operator node within the graph.
        *   `name`: A human-readable name for the operator (e.g., "conv_layer1", "activation_relu3").
        *   `op_type`: A string indicating the type of operation (e.g., "Conv2D", "ReLU", "MatMul").
        *   `inputs`: A list of `TensorNode`s that are inputs to this operation.
        *   `outputs`: A list of `TensorNode`s that are outputs from this operation.
        *   `attributes`: A dictionary containing operation-specific parameters (e.g., stride and padding for a convolution, axis for a reduction).
