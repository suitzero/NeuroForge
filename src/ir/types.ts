/**
 * Unique identifier for a Tensor within the computation graph.
 */
export type TensorId = string | number;

/**
 * Unique identifier for an Operation within the computation graph.
 */
export type OperatorId = string | number;

/**
 * Represents the data type of a Tensor's elements.
 * TODO: Expand with more specific types (e.g., 'float16', 'int8', etc.)
 */
export type DataType = 'float32' | 'int32' | 'boolean';

/**
 * Represents the shape of a Tensor (dimensions).
 */
export type Shape = readonly number[];
