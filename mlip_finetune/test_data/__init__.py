"""Test data for mlip-finetune package."""
from pathlib import Path

# Path to test data directory
TEST_DATA_DIR = Path(__file__).parent

# Available test datasets
BTO_100 = TEST_DATA_DIR / "bto_100.xyz"


def get_test_data_path(name: str = "bto_100") -> Path:
    """Get path to test dataset.
    
    Args:
        name: Name of test dataset (default: "bto_100")
        
    Returns:
        Path to the test data file
    """
    data_map = {
        "bto_100": BTO_100,
    }
    
    if name not in data_map:
        raise ValueError(f"Unknown test dataset: {name}. Available: {list(data_map.keys())}")
    
    return data_map[name]
