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

        # Ensure all input/output tensors are declared in the graph before adding operator
        for tensor_id in operator.input_tensor_ids + operator.output_tensor_ids:
            if tensor_id not in self.tensors:
                raise ValueError(f"Tensor with id {tensor_id} (referenced by operator {operator.id}) not found in graph. Declare all tensors first.")

        self.operators[operator.id] = operator

        # Link producer/consumer relationships
        for tensor_id in operator.input_tensor_ids:
            self.tensors[tensor_id].consumers.append(operator.id)

        for tensor_id in operator.output_tensor_ids:
            if self.tensors[tensor_id].producer is not None and self.tensors[tensor_id].producer != operator.id:
                 raise ValueError(f"Tensor {tensor_id} already has a producer {self.tensors[tensor_id].producer}, cannot be produced by {operator.id}")
            self.tensors[tensor_id].producer = operator.id

        self._nodes_in_topological_order = [] # Invalidate previous sort

    def get_tensor(self, id: TensorId) -> TensorNode:
        if id not in self.tensors:
            raise KeyError(f"Tensor with id {id} not found.")
        return self.tensors[id]

    def get_operator(self, id: OperatorId) -> OperatorNode:
        if id not in self.operators:
            raise KeyError(f"Operator with id {id} not found.")
        return self.operators[id]

    def get_tensor_by_name(self, name: str) -> Optional[TensorNode]:
        for tensor in self.tensors.values():
            if tensor.name == name:
                return tensor
        return None

    def get_operator_by_name(self, name: str) -> Optional[OperatorNode]:
        for op in self.operators.values():
            if op.name == name:
                return op
        return None

    def remove_operator(self, operator_id: OperatorId):
        if operator_id not in self.operators:
            print(f"Warning: Operator with id {operator_id} not found. Cannot remove.")
            return

        op_to_remove = self.operators[operator_id]

        # Unlink from input tensors' consumer lists
        for tensor_id in op_to_remove.input_tensor_ids:
            if tensor_id in self.tensors:
                if operator_id in self.tensors[tensor_id].consumers:
                    self.tensors[tensor_id].consumers.remove(operator_id)

        # Clear producer for output tensors produced by this op
        # Note: This doesn't automatically remove the output tensors themselves.
        # Orphaned tensors might need separate handling (e.g., a graph cleanup pass).
        for tensor_id in op_to_remove.output_tensor_ids:
            if tensor_id in self.tensors:
                if self.tensors[tensor_id].producer == operator_id:
                    self.tensors[tensor_id].producer = None
                    # print(f"Info: Tensor {tensor_id} is now orphaned as its producer {operator_id} was removed.")


        del self.operators[operator_id]
        self._nodes_in_topological_order = [] # Invalidate previous sort
        print(f"Operator {operator_id} removed from graph.")


    def get_graph_inputs(self) -> List[TensorNode]:
        return [self.get_tensor(id) for id in self.input_ids]

    def get_graph_outputs(self) -> List[TensorNode]:
        return [self.get_tensor(id) for id in self.output_ids]

    def topologically_sort_operators(self) -> List[OperatorId]:
        """
        Performs a topological sort of the operators using Kahn's algorithm.
        If the graph has cycles, this will not include all nodes.
        A more robust implementation would detect and report cycles.
        """
        if self._nodes_in_topological_order and len(self._nodes_in_topological_order) == len(self.operators):
            # Already sorted and full, return cached
            return self._nodes_in_topological_order

        print("Attempting topological sort using Kahn's algorithm...")

        in_degree: Dict[OperatorId, int] = {op_id: 0 for op_id in self.operators}
        # Adjacency list: op_id -> list of ops that depend on it
        adj: Dict[OperatorId, List[OperatorId]] = {op_id: [] for op_id in self.operators}

        for op_id, op_node in self.operators.items():
            # For each output tensor of this op
            for output_tensor_id in op_node.output_tensor_ids:
                tensor = self.tensors[output_tensor_id]
                # For each consumer of this output tensor
                for consumer_op_id in tensor.consumers:
                    if consumer_op_id in self.operators: # Ensure consumer is part of the current graph
                        adj[op_id].append(consumer_op_id)
                        in_degree[consumer_op_id] += 1

        queue: List[OperatorId] = [op_id for op_id, degree in in_degree.items() if degree == 0]
        sorted_list: List[OperatorId] = []

        while queue:
            u = queue.pop(0)
            sorted_list.append(u)
            for v_op_id in adj[u]:
                in_degree[v_op_id] -= 1
                if in_degree[v_op_id] == 0:
                    queue.append(v_op_id)

        if len(sorted_list) != len(self.operators):
            print(f"Warning: Topological sort incomplete. Graph might have a cycle or disconnected components. Sorted {len(sorted_list)} of {len(self.operators)} operators.")
            # Fallback to simple list if sort is incomplete to avoid breaking other parts for now
            # A real compiler would likely raise an error here.
            self._nodes_in_topological_order = list(self.operators.keys())
        else:
            self._nodes_in_topological_order = sorted_list
            print("Topological sort successful.")

        return self._nodes_in_topological_order


    def is_valid(self) -> bool:
        """Performs basic integrity checks on the graph."""
        valid = True
        print("Performing graph validation...")

        # Check tensor existence for all operator inputs/outputs
        for op_id, op_node in self.operators.items():
            for tensor_id in op_node.input_tensor_ids + op_node.output_tensor_ids:
                if tensor_id not in self.tensors:
                    print(f"Error: Operator {op_id} references non-existent tensor {tensor_id}.")
                    valid = False

        # Check producer/consumer consistency
        for tensor_id, tensor_node in self.tensors.items():
            if tensor_node.producer:
                producer_op_id = tensor_node.producer
                if producer_op_id not in self.operators:
                    print(f"Error: Tensor {tensor_id} has non-existent producer {producer_op_id}.")
                    valid = False
                elif tensor_id not in self.operators[producer_op_id].output_tensor_ids:
                    print(f"Error: Tensor {tensor_id} producer {producer_op_id} does not list it as an output.")
                    valid = False

            for consumer_op_id in tensor_node.consumers:
                if consumer_op_id not in self.operators:
                    print(f"Error: Tensor {tensor_id} has non-existent consumer {consumer_op_id}.")
                    valid = False
                elif tensor_id not in self.operators[consumer_op_id].input_tensor_ids:
                    print(f"Error: Tensor {tensor_id} consumer {consumer_op_id} does not list it as an input.")
                    valid = False

        # Check graph input/output tensor existence
        for tensor_id in self.input_ids + self.output_ids:
            if tensor_id not in self.tensors:
                print(f"Error: Graph input/output tensor {tensor_id} not found in graph tensors.")
                valid = False

        if valid:
            print("Graph validation successful.")
        else:
            print("Graph validation failed.")
        return valid

    def __repr__(self) -> str:
        return f"Graph(name='{self.name}', {len(self.tensors)} tensors, {len(self.operators)} operators)"
