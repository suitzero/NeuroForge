from ir import Graph # Assuming ir.py is in the same directory or accessible via PYTHONPATH
from hal import HardwareAbstractionLayer # Ensure HAL is also importable

class CodeGenerator:
    def __init__(self, hal: HardwareAbstractionLayer):
        self.hal = hal
        print(f"Code Generator initialized (IR-aware) with HAL for: {self.hal.target_hardware}")

    def generate_code(self, graph: Graph) -> str:
        """
        Placeholder for generating hardware-specific code from an IR Graph.
        Returns the path to the generated code/binary.
        """
        print(f"Generating code for {self.hal.target_hardware} based on IR Graph: {graph.name}...")

        if not graph.operators:
            print("Warning: Graph has no operators, nothing to generate.")
            return f"./output_empty_{self.hal.target_hardware}.bin"

        # In a real implementation, this would traverse the graph (likely using
        # the topologically sorted order of operators) and translate each operator
        # into hardware-specific instructions using the HAL.

        print(f"Graph has {len(graph.operators)} operators to generate code for.")
        # Example: Iterate through operators (no actual code generation)
        for op_id in graph.topologically_sort_operators():
            operator = graph.get_operator(op_id)
            print(f"  Processing operator for code generation: {operator.name} ({operator.op_type})")
            # Here, you would use self.hal to get target-specific info
            # and generate instructions.

        output_path = f"./output_{graph.name}_{self.hal.target_hardware}.bin"
        print(f"Code generation complete (placeholder). Output artifact at: {output_path}")
        return output_path

def get_code_generator(hal: HardwareAbstractionLayer) -> CodeGenerator:
    """Factory function to get a CodeGenerator instance."""
    return CodeGenerator(hal)
