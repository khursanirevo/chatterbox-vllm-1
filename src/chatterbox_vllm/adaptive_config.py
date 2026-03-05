"""
Adaptive configuration system for vLLM TTS optimization.

Implements request-aware routing with 3 length categories, each optimized
for different goals:
- Short (<20 tokens): Interactive, highest priority, TTFA < 1s
- Medium (20-50 tokens): Standard priority, TTFA < 2s
- Long (>50 tokens): Background priority, TTFA < 4s
"""

from typing import Dict, Any, Optional
import os
from dataclasses import dataclass


@dataclass
class AdaptiveProfile:
    """Configuration profile for a specific request category."""
    name: str
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    enable_prefix_caching: bool
    enforce_eager: bool
    gpu_memory_utilization: float
    description: str


# Short requests (<20 tokens) - Speed Optimized
# Target: TTFA < 1s, highest priority for interactive use
PROFILE_SHORT = AdaptiveProfile(
    name="short",
    max_model_len=256,
    max_num_seqs=16,
    max_num_batched_tokens=4096,
    enable_prefix_caching=True,
    enforce_eager=True,  # Avoid CUDA graph compilation overhead
    gpu_memory_utilization=0.02,
    description="Interactive requests prioritized for speed (TTFA < 1s)"
)

# Medium requests (20-50 tokens) - Balanced
# Target: TTFA < 2s, good balance of speed and throughput
PROFILE_MEDIUM = AdaptiveProfile(
    name="medium",
    max_model_len=512,
    max_num_seqs=8,
    max_num_batched_tokens=6144,
    enable_prefix_caching=True,
    enforce_eager=True,
    gpu_memory_utilization=0.03,
    description="Standard requests with balanced configuration (TTFA < 2s)"
)

# Long requests (>50 tokens) - Throughput Optimized
# Target: TTFA < 4s, optimized for throughput
PROFILE_LONG = AdaptiveProfile(
    name="long",
    max_model_len=1000,
    max_num_seqs=4,
    max_num_batched_tokens=8192,
    enable_prefix_caching=True,
    enforce_eager=False,  # Can use CUDA graphs for efficiency
    gpu_memory_utilization=0.04,
    description="Background requests optimized for throughput (TTFA < 4s)"
)


# Profile registry
PROFILES: Dict[str, AdaptiveProfile] = {
    "short": PROFILE_SHORT,
    "medium": PROFILE_MEDIUM,
    "long": PROFILE_LONG,
}

# Token thresholds for classification
SHORT_THRESHOLD_TOKENS = 20
MEDIUM_THRESHOLD_TOKENS = 50


def classify_request(text: str, tokenizer, variant: str = "english") -> str:
    """
    Classify a request into short/medium/long category.

    Args:
        text: Input text to classify
        tokenizer: Tokenizer to count tokens
        variant: Model variant ("english" or "multilingual")

    Returns:
        Category string: "short", "medium", or "long"
    """
    from chatterbox_vllm.text_utils import punc_norm

    # Normalize text same as generation
    normalized_text = "[START]" + punc_norm(text) + "[STOP]"
    if variant == "multilingual":
        normalized_text = f"<en>{normalized_text}"

    # Tokenize to count tokens
    tokens = tokenizer.encode(normalized_text)
    token_count = len(tokens)

    if token_count < SHORT_THRESHOLD_TOKENS:
        return "short"
    elif token_count < MEDIUM_THRESHOLD_TOKENS:
        return "medium"
    else:
        return "long"


def classify_request_by_chars(text: str) -> str:
    """
    Fast classification based on character count (heuristic).

    This is a lightweight approximation when tokenizer is not available.
    Approximate ratios:
    - English: ~4 chars per token
    - Multilingual: ~3 chars per token

    Args:
        text: Input text to classify

    Returns:
        Category string: "short", "medium", or "long"
    """
    char_count = len(text)

    # Heuristic: ~4 chars per token for English
    estimated_tokens = char_count / 4

    if estimated_tokens < SHORT_THRESHOLD_TOKENS:
        return "short"
    elif estimated_tokens < MEDIUM_THRESHOLD_TOKENS:
        return "medium"
    else:
        return "long"


def get_profile(category: str) -> AdaptiveProfile:
    """
    Get configuration profile for a category.

    Args:
        category: Category name ("short", "medium", "long")

    Returns:
        AdaptiveProfile with configuration parameters

    Raises:
        ValueError: If category is not recognized
    """
    if category not in PROFILES:
        raise ValueError(f"Unknown category '{category}'. Must be one of: {list(PROFILES.keys())}")

    return PROFILES[category]


def get_profile_params(category: str) -> Dict[str, Any]:
    """
    Get vLLM engine parameters for a category.

    Returns a dictionary suitable for passing to AsyncEngineArgs.

    Args:
        category: Category name ("short", "medium", "long")

    Returns:
        Dictionary of vLLM engine parameters
    """
    profile = get_profile(category)

    return {
        "max_model_len": profile.max_model_len,
        "gpu_memory_utilization": profile.gpu_memory_utilization,
        "enforce_eager": profile.enforce_eager,
        # Note: max_num_seqs and max_num_batched_tokens are set via
        # environment variables or engine args in vLLM
    }


def get_priority_for_category(category: str) -> int:
    """
    Get vLLM scheduling priority for a category.

    Lower number = higher priority (vLLM convention).

    Args:
        category: Category name

    Returns:
        Priority value (0=highest, 9=lowest)
    """
    priorities = {
        "short": 0,   # Highest priority
        "medium": 5,  # Medium priority
        "long": 9,    # Lowest priority
    }
    return priorities.get(category, 5)


# Feature flag to enable/disable adaptive mode
ADAPTIVE_MODE_ENABLED = os.getenv("CHATTERBOX_ADAPTIVE_MODE", "true").lower() == "true"


def is_adaptive_mode_enabled() -> bool:
    """Check if adaptive mode is enabled."""
    return ADAPTIVE_MODE_ENABLED


def enable_adaptive_mode():
    """Enable adaptive mode."""
    global ADAPTIVE_MODE_ENABLED
    ADAPTIVE_MODE_ENABLED = True


def disable_adaptive_mode():
    """Disable adaptive mode (revert to single-engine baseline)."""
    global ADAPTIVE_MODE_ENABLED
    ADAPTIVE_MODE_ENABLED = False


def get_default_profile() -> str:
    """
    Get the default profile to use when adaptive mode is disabled.

    Returns:
        Default category name
    """
    return os.getenv("CHATTERBOX_DEFAULT_PROFILE", "medium")


def print_profile_summary():
    """Print a summary of available profiles."""
    print("\n" + "="*80)
    print("ADAPTIVE CONFIGURATION PROFILES")
    print("="*80)

    for category, profile in PROFILES.items():
        print(f"\n{category.upper()} Requests:")
        print(f"  Description: {profile.description}")
        print(f"  Max Model Length: {profile.max_model_len} tokens")
        print(f"  Max Concurrent Sequences: {profile.max_num_seqs}")
        print(f"  Max Batched Tokens: {profile.max_num_batched_tokens}")
        print(f"  GPU Memory Utilization: {profile.gpu_memory_utilization:.3f}")
        print(f"  Enforce Eager: {profile.enforce_eager}")
        print(f"  Priority: {get_priority_for_category(category)}")

    print(f"\nAdaptive Mode: {'ENABLED' if is_adaptive_mode_enabled() else 'DISABLED'}")
    if not is_adaptive_mode_enabled():
        print(f"Default Profile: {get_default_profile()}")

    print("="*80 + "\n")
