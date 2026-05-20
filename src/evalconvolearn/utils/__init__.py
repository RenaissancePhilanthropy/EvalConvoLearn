"""Utility modules for evalconvolearn benchmarks."""

# Alignment matrices
from .alignment_matrices import (
    generate_learning_alignment_matrix,
    generate_placement_test_alignment_matrix,
)

# Conversation runner
from .conversation_utils import run_conversation_to_completion

# Metric calculation functions
from .benchmark_metrics import (
    calculate_placement_test_alignment,
)
from .benchmark_results import (
    PlacementTestResult,
    print_lfc_results,
    print_mcp_results,
    print_placement_results,
    print_placement_test_results,
)

# Data loaders
from .data_loaders import (
    get_benchmark_output_dir,
    get_data_dir,
    get_florida_doe_data_dir,
    get_tutor_responses_csv_path,
    load_tagged_skill_ids,
    load_tutor_responses_mapping,
    render_conversation_messages,
)

# Learner configuration data
from .learner_configs import (
    get_beginner_config,
    get_blank_config,
    get_expert_config,
    get_intermediate_config,
    get_placement_test_skill_levels,
)

__all__ = [
    # Conversation runner
    "run_conversation_to_completion",
    # Result classes
    "PlacementTestResult",
    # Result printers
    "print_placement_test_results",
    "print_placement_results",
    "print_mcp_results",
    "print_lfc_results",
    # Metrics
    "calculate_placement_test_alignment",
    # Learner configs
    "get_placement_test_skill_levels",
    "get_beginner_config",
    "get_intermediate_config",
    "get_expert_config",
    "get_blank_config",
    # Data loaders
    "get_data_dir",
    "get_florida_doe_data_dir",
    "load_tagged_skill_ids",
    "get_tutor_responses_csv_path",
    "load_tutor_responses_mapping",
    "get_benchmark_output_dir",
    "render_conversation_messages",
    # Alignment matrices
    "generate_placement_test_alignment_matrix",
    "generate_learning_alignment_matrix",
]
