import { TensorId, OperatorId, DataType, Shape } from './types';

// Placeholder for future Tensor metadata and potential data storage
export interface Tensor {
  id: TensorId;
  name?: string;
  dtype: DataType;
  shape: Shape;
  // data?: ArrayBufferView; // Actual data, might be managed differently
  producerOpId?: OperatorId; // Operation that produces this tensor
  consumerOpIds?: OperatorId[]; // Operations that consume this tensor
}

// Placeholder for an operation in the computation graph
export interface Operation {
  id: OperatorId;
  opType: string; // e.g., 'add', 'matmul', 'conv2d'
  name?: string;
  inputTensorIds: TensorId[];
  outputTensorIds: TensorId[];
  attributes?: Record<string, any>; // Operation-specific attributes
}

// Placeholder for the computation graph
export interface ComputationGraph {
  name?: string;
  inputs: TensorId[];
  outputs: TensorId[];
  tensors: Map<TensorId, Tensor>;
  operations: Map<OperatorId, Operation>;
  // Could include methods for adding nodes, validation, topological sort etc.
}

// Example of how one might create a graph (very basic)
export function createGraph(name?: string): ComputationGraph {
  return {
    name,
    inputs: [],
    outputs: [],
    tensors: new Map<TensorId, Tensor>(),
    operations: new Map<OperatorId, Operation>(),
  };
}
