class HardwareAbstractionLayer:
    def __init__(self, target_hardware: str):
        self.target_hardware = target_hardware
        print(f"Hardware Abstraction Layer initialized for: {self.target_hardware}")

    def get_hardware_capabilities(self) -> dict:
        """
        Placeholder for querying hardware capabilities.
        Returns a dictionary of hardware features.
        """
        print(f"Querying capabilities for {self.target_hardware}...")
        # Dummy capabilities
        capabilities = {"supported_ops": ["conv2d", "relu", "dense"], "memory_gb": 2}
        print(f"Capabilities received: {capabilities}")
        return capabilities

def get_hal(target_hardware: str) -> HardwareAbstractionLayer:
    """Factory function to get a HAL instance."""
    return HardwareAbstractionLayer(target_hardware)
