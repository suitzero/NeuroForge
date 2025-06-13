from typing import List, Tuple, Dict, Optional, NewType

# Using NewType for better type hinting of IDs, could also be simple strings or ints
TensorId = NewType('TensorId', str)
OperatorId = NewType('OperatorId', str)

class TensorNode:
    """Represents a data tensor within the graph."""
    def __init__(self,
                 id: TensorId,
                 name: str,
                 shape: Tuple[int, ...],
                 dtype: str,
                 producer: Optional[OperatorId] = None):
        self.id: TensorId = id
        self.name: str = name
        self.shape: Tuple[int, ...] = shape
        self.dtype: str = dtype
        self.producer: Optional[OperatorId] = producer # ID of the OperatorNode that produces this tensor
        self.consumers: List[OperatorId] = [] # IDs of OperatorNodes that consume this tensor

    def __repr__(self) -> str:
        return f"TensorNode(id='{self.id}', name='{self.name}', shape={self.shape}, dtype='{self.dtype}')"

class OperatorNode:
    """Represents a computational operation in the model."""
    def __init__(self,
                 id: OperatorId,
                 name: str,
                 op_type: str,
                 input_tensor_ids: List[TensorId],
                 output_tensor_ids: List[TensorId],
                 attributes: Optional[Dict] = None):
        self.id: OperatorId = id
        self.name: str = name
        self.op_type: str = op_type
        self.input_tensor_ids: List[TensorId] = input_tensor_ids # IDs of input TensorNodes
        self.output_tensor_ids: List[TensorId] = output_tensor_ids # IDs of output TensorNodes
        self.attributes: Dict = attributes if attributes is not None else {}

    def __repr__(self) -> str:
        return f"OperatorNode(id='{self.id}', name='{self.name}', op_type='{self.op_type}')"

class Graph:
    """The main container for the entire model representation."""
    def __init__(self,
                 name: str,
                 input_ids: List[TensorId],
                 output_ids: List[TensorId]):
        self.name: str = name
        self.tensors: Dict[TensorId, TensorNode] = {}
        self.operators: Dict[OperatorId, OperatorNode] = {}
        self.input_ids: List[TensorId] = input_ids # IDs of primary graph input tensors
        self.output_ids: List[TensorId] = output_ids # IDs of primary graph output tensors
        self._nodes_in_topological_order: List[OperatorId] = [] # Store operator IDs

    def add_tensor(self, tensor: TensorNode):
        if tensor.id in self.tensors:
            raise ValueError(f"Tensor with id {tensor.id} already exists in the graph.")
        self.tensors[tensor.id] = tensor

    def add_operator(self, operator: OperatorNode):
        if operator.id in self.operators:
            raise ValueError(f"Operator with id {operator.id} already exists in the graph.")
        self.operators[operator.id] = operator

        # Link producer/consumer relationships
        for tensor_id in operator.input_tensor_ids:
            if tensor_id not in self.tensors:
                raise ValueError(f"Input tensor with id {tensor_id} for operator {operator.id} not found in graph.")
            self.tensors[tensor_id].consumers.append(operator.id)

        for tensor_id in operator.output_tensor_ids:
            if tensor_id not in self.tensors:
                # Output tensors might be created by this op, so we might add them here
                # For now, assume they are pre-declared or handle appropriately in parser
                pass # Or raise ValueError if they must pre-exist
            self.tensors[tensor_id].producer = operator.id


    def get_tensor(self, id: TensorId) -> TensorNode:
        if id not in self.tensors:
            raise KeyError(f"Tensor with id {id} not found.")
        return self.tensors[id]

    def get_operator(self, id: OperatorId) -> OperatorNode:
        if id not in self.operators:
            raise KeyError(f"Operator with id {id} not found.")
        return self.operators[id]

    def get_graph_inputs(self) -> List[TensorNode]:
        return [self.get_tensor(id) for id in self.input_ids]

    def get_graph_outputs(self) -> List[TensorNode]:
        return [self.get_tensor(id) for id in self.output_ids]

    def topologically_sort_operators(self) -> List[OperatorId]:
        """
        Performs a topological sort of the operators.
        (Placeholder - a real implementation would use Kahn's algorithm or DFS)
        For now, just returns operators in order of addition if not already sorted.
        This is a complex task and would require a full graph traversal implementation.
        """
        if self._nodes_in_topological_order:
            return self._nodes_in_topological_order

        # Placeholder: This is NOT a real topological sort.
        # A real implementation would build the graph and then sort.
        # For now, we'll assume operators are added in a somewhat valid order
        # or this needs to be explicitly called after graph construction.
        print("Warning: Using placeholder topological sort. Operator order may not be correct.")
        self._nodes_in_topological_order = list(self.operators.keys())
        return self._nodes_in_topological_order


    def __repr__(self) -> str:
        return f"Graph(name='{self.name}', {len(self.tensors)} tensors, {len(self.operators)} operators)"
