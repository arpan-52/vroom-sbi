"""
Weight augmentation strategies for training robust models.

Implements various missing channel patterns to teach the network interpolation:
- Random scattered missing channels
- Contiguous gaps (RFI simulation)
- Noise variation
- Large RFI blocks
"""

import numpy as np


def augment_weights_scattered(weights: np.ndarray, missing_prob: float = 0.1) -> np.ndarray:
    """
    Randomly set channels to missing with given probability.
    
    Parameters
    ----------
    weights : np.ndarray
        Original channel weights
    missing_prob : float
        Probability that each channel is set to 0 (missing)
        
    Returns
    -------
    np.ndarray
        Augmented weights
    """
    aug_weights = weights.copy()
    mask = np.random.random(len(weights)) < missing_prob
    aug_weights[mask] = 0.0
    return aug_weights


def augment_weights_contiguous_gap(weights: np.ndarray, 
                                   gap_prob: float = 0.3,
                                   min_gap: int = 2, 
                                   max_gap: int = 8) -> np.ndarray:
    """
    Create a random contiguous gap to simulate RFI.
    
    Parameters
    ----------
    weights : np.ndarray
        Original channel weights
    gap_prob : float
        Probability of creating a gap
    min_gap : int
        Minimum gap size in channels
    max_gap : int
        Maximum gap size in channels
        
    Returns
    -------
    np.ndarray
        Augmented weights with contiguous gap
    """
    aug_weights = weights.copy()
    
    if np.random.random() < gap_prob and len(weights) > max_gap:
        gap_size = np.random.randint(min_gap, max_gap + 1)
        start_idx = np.random.randint(0, len(weights) - gap_size + 1)
        aug_weights[start_idx:start_idx + gap_size] = 0.0
    
    return aug_weights


def augment_weights_large_rfi_block(weights: np.ndarray,
                                     block_prob: float = 0.1,
                                     min_block: int = 10,
                                     max_block: int = 30) -> np.ndarray:
    """
    Create a large contiguous gap to simulate large RFI blocks.
    
    Parameters
    ----------
    weights : np.ndarray
        Original channel weights
    block_prob : float
        Probability of creating a large block
    min_block : int
        Minimum block size in channels
    max_block : int
        Maximum block size in channels
        
    Returns
    -------
    np.ndarray
        Augmented weights with large RFI block
    """
    aug_weights = weights.copy()
    
    if np.random.random() < block_prob and len(weights) > max_block:
        block_size = np.random.randint(min_block, min(max_block + 1, len(weights) // 2))
        start_idx = np.random.randint(0, len(weights) - block_size + 1)
        aug_weights[start_idx:start_idx + block_size] = 0.0
    
    return aug_weights


def augment_weights_noise_variation(weights: np.ndarray, 
                                    variation_scale: float = 0.2) -> np.ndarray:
    """
    Add variation to non-zero weights to simulate varying noise levels.
    
    Parameters
    ----------
    weights : np.ndarray
        Original channel weights
    variation_scale : float
        Scale of variation to add (relative to weight value)
        
    Returns
    -------
    np.ndarray
        Augmented weights with noise variation
    """
    aug_weights = weights.copy()
    
    # Only vary non-zero weights
    non_zero_mask = aug_weights > 0
    variation = np.random.normal(0, variation_scale, len(aug_weights))
    aug_weights[non_zero_mask] = aug_weights[non_zero_mask] * (1 + variation[non_zero_mask])
    
    # Clip to valid range [0, 1+small tolerance]
    aug_weights = np.clip(aug_weights, 0.0, 1.5)
    
    # Normalize so max is still 1.0
    if np.max(aug_weights) > 0:
        aug_weights = aug_weights / np.max(aug_weights)
    
    return aug_weights


def augment_weights_combined(weights: np.ndarray,
                            scattered_prob: float = 0.3,
                            gap_prob: float = 0.3,
                            large_block_prob: float = 0.1,
                            noise_variation: bool = True) -> np.ndarray:
    """
    Apply random combination of augmentation strategies.
    
    This is the main augmentation function to use during training.
    
    Parameters
    ----------
    weights : np.ndarray
        Original channel weights
    scattered_prob : float
        Probability of applying scattered missing augmentation
    gap_prob : float
        Probability of applying contiguous gap augmentation
    large_block_prob : float
        Probability of applying large RFI block augmentation
    noise_variation : bool
        Whether to apply noise variation
        
    Returns
    -------
    np.ndarray
        Augmented weights
    """
    aug_weights = weights.copy()
    
    # Apply scattered missing
    if np.random.random() < scattered_prob:
        aug_weights = augment_weights_scattered(aug_weights, missing_prob=0.1)
    
    # Apply contiguous gap
    if np.random.random() < gap_prob:
        aug_weights = augment_weights_contiguous_gap(aug_weights)
    
    # Apply large RFI block
    if np.random.random() < large_block_prob:
        aug_weights = augment_weights_large_rfi_block(aug_weights)
    
    # Apply noise variation
    if noise_variation:
        aug_weights = augment_weights_noise_variation(aug_weights)
    
    return aug_weights


def generate_augmented_weights_batch(base_weights: np.ndarray, 
                                     batch_size: int) -> np.ndarray:
    """
    Generate a batch of augmented weights for training.
    
    Parameters
    ----------
    base_weights : np.ndarray
        Original channel weights
    batch_size : int
        Number of augmented weight arrays to generate
        
    Returns
    -------
    np.ndarray
        Array of shape (batch_size, n_channels) with augmented weights
    """
    augmented_batch = np.zeros((batch_size, len(base_weights)))
    
    for i in range(batch_size):
        augmented_batch[i] = augment_weights_combined(base_weights)
    
    return augmented_batch
