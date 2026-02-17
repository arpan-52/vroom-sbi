"""
Data loading utilities for VROOM-SBI training.

Handles:
- Simulation dataset creation
- Streaming from HDF5 (memory-efficient)
- Async prefetching: Disk → RAM → GPU pipeline
- Saving/loading simulations
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
import logging
import threading
import queue

logger = logging.getLogger(__name__)


class HDF5StreamingDataset(IterableDataset):
    """
    Streaming dataset that reads from HDF5 file.
    
    Memory usage: O(batch_size * prefetch_factor), not O(dataset_size)
    
    Parameters
    ----------
    h5_path : Path
        Path to HDF5 file with 'theta' and 'x' datasets
    batch_size : int
        Batch size for reading chunks
    shuffle : bool
        Whether to shuffle indices (loads index array only)
    device : str
        Target device for tensors
    """
    
    def __init__(
        self,
        h5_path: Path,
        batch_size: int = 256,
        shuffle: bool = True,
        device: str = 'cuda',
    ):
        self.h5_path = Path(h5_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        
        # Read metadata without loading data
        import h5py
        with h5py.File(self.h5_path, 'r') as f:
            self.n_samples = f['theta'].shape[0]
            self.n_params = f['theta'].shape[1]
            self.x_dim = f['x'].shape[1]
            self.n_freq = f.attrs.get('n_freq', self.x_dim // 2)
    
    def __len__(self):
        return self.n_samples
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over batches, reading from disk."""
        import h5py
        
        # Generate indices
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Open file for this worker
        with h5py.File(self.h5_path, 'r') as f:
            theta_ds = f['theta']
            x_ds = f['x']
            
            # Yield batches
            for start in range(0, self.n_samples, self.batch_size):
                end = min(start + self.batch_size, self.n_samples)
                batch_indices = indices[start:end]
                
                # Sort indices for efficient HDF5 read (contiguous access)
                sorted_idx = np.argsort(batch_indices)
                sorted_batch_indices = batch_indices[sorted_idx]
                
                # Read from disk
                theta_batch = theta_ds[sorted_batch_indices]
                x_batch = x_ds[sorted_batch_indices]
                
                # Unsort to restore shuffle order
                unsort_idx = np.argsort(sorted_idx)
                theta_batch = theta_batch[unsort_idx]
                x_batch = x_batch[unsort_idx]
                
                # Convert to tensors and move to device
                theta_t = torch.tensor(theta_batch, dtype=torch.float32, device=self.device)
                x_t = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
                
                yield theta_t, x_t


class AsyncPrefetchLoader:
    """
    Async data loader that prefetches batches in background thread.
    
    Pipeline: Disk → RAM (background) → GPU (foreground)
    
    Parameters
    ----------
    h5_path : Path
        Path to HDF5 file
    batch_size : int
        Batch size
    prefetch_batches : int
        Number of batches to prefetch (controls RAM usage)
    shuffle : bool
        Whether to shuffle
    device : str
        Target GPU device
    """
    
    def __init__(
        self,
        h5_path: Path,
        batch_size: int = 256,
        prefetch_batches: int = 4,
        shuffle: bool = True,
        device: str = 'cuda',
    ):
        self.h5_path = Path(h5_path)
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.shuffle = shuffle
        self.device = device
        
        # Read metadata
        import h5py
        with h5py.File(self.h5_path, 'r') as f:
            self.n_samples = f['theta'].shape[0]
            self.n_params = f['theta'].shape[1]
            self.x_dim = f['x'].shape[1]
        
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
    
    def __len__(self):
        return self.n_batches
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate with async prefetching."""
        import h5py
        
        # Queue for prefetched batches
        prefetch_queue = queue.Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        
        # Generate indices
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        def prefetch_worker():
            """Background thread that reads from disk to RAM."""
            try:
                with h5py.File(self.h5_path, 'r') as f:
                    theta_ds = f['theta']
                    x_ds = f['x']
                    
                    for start in range(0, self.n_samples, self.batch_size):
                        if stop_event.is_set():
                            break
                        
                        end = min(start + self.batch_size, self.n_samples)
                        batch_indices = indices[start:end]
                        
                        # Sort for efficient HDF5 read
                        sorted_idx = np.argsort(batch_indices)
                        sorted_batch_indices = batch_indices[sorted_idx]
                        
                        # Read from disk to RAM (numpy arrays)
                        theta_batch = theta_ds[sorted_batch_indices]
                        x_batch = x_ds[sorted_batch_indices]
                        
                        # Unsort
                        unsort_idx = np.argsort(sorted_idx)
                        theta_batch = theta_batch[unsort_idx]
                        x_batch = x_batch[unsort_idx]
                        
                        # Put in queue (blocks if queue full)
                        prefetch_queue.put((theta_batch, x_batch))
                
                # Signal end of data
                prefetch_queue.put(None)
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                prefetch_queue.put(None)
        
        # Start prefetch thread
        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()
        
        try:
            # Consume from queue, move to GPU
            while True:
                item = prefetch_queue.get()
                if item is None:
                    break
                
                theta_batch, x_batch = item
                
                # Move to GPU (this is the only GPU transfer)
                theta_t = torch.tensor(theta_batch, dtype=torch.float32, device=self.device)
                x_t = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
                
                yield theta_t, x_t
        finally:
            stop_event.set()
            prefetch_thread.join(timeout=1.0)


def create_streaming_loader(
    h5_path: Path,
    batch_size: int = 256,
    prefetch_batches: int = 4,
    shuffle: bool = True,
    device: str = 'cuda',
) -> AsyncPrefetchLoader:
    """
    Create an async streaming data loader for training.
    
    Parameters
    ----------
    h5_path : Path
        Path to HDF5 simulation file
    batch_size : int
        Training batch size
    prefetch_batches : int
        Number of batches to prefetch (RAM usage = prefetch_batches * batch_size * data_size)
    shuffle : bool
        Whether to shuffle data each epoch
    device : str
        Target device
        
    Returns
    -------
    AsyncPrefetchLoader
        Streaming data loader
    """
    return AsyncPrefetchLoader(
        h5_path=h5_path,
        batch_size=batch_size,
        prefetch_batches=prefetch_batches,
        shuffle=shuffle,
        device=device,
    )


class SimulationDataset(Dataset):
    """
    Dataset for training with simulated spectra.
    
    Parameters
    ----------
    spectra : np.ndarray
        Array of shape (n_samples, 2 * n_freq) containing [Q, U]
    weights : np.ndarray
        Array of shape (n_samples, n_freq) containing channel weights
    theta : np.ndarray, optional
        Array of shape (n_samples, n_params) containing parameters
    labels : np.ndarray, optional
        Array of shape (n_samples,) containing class labels
    """
    
    def __init__(
        self,
        spectra: np.ndarray,
        weights: np.ndarray,
        theta: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ):
        self.spectra = torch.tensor(spectra, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        if theta is not None:
            self.theta = torch.tensor(theta, dtype=torch.float32)
        else:
            self.theta = None
            
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None
        
        self.n_samples = len(spectra)
        self.n_freq = weights.shape[1] if weights.ndim > 1 else len(weights)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns
        -------
        dict
            Dictionary with 'x' (spectra + weights), 'theta' (if available),
            'label' (if available)
        """
        # Concatenate [Q, U, weights] for classifier input
        x = torch.cat([self.spectra[idx], self.weights[idx]], dim=0)
        
        result = {'x': x, 'spectra': self.spectra[idx], 'weights': self.weights[idx]}
        
        if self.theta is not None:
            result['theta'] = self.theta[idx]
        
        if self.labels is not None:
            result['label'] = self.labels[idx]
        
        return result


def save_simulations(
    save_path: Path,
    spectra: np.ndarray,
    weights: np.ndarray,
    theta: np.ndarray,
    n_components: int,
    model_type: str,
    n_freq: int,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save simulations using torch.save().
    
    Parameters
    ----------
    save_path : Path
        Output path (.pt file)
    spectra : np.ndarray
        Simulated spectra (n_samples, 2*n_freq)
    weights : np.ndarray
        Channel weights (n_samples, n_freq)
    theta : np.ndarray
        Parameters (n_samples, n_params)
    n_components : int
        Number of components
    model_type : str
        Physical model type
    n_freq : int
        Number of frequency channels
    metadata : dict, optional
        Additional metadata to save
    """
    save_dict = {
        'spectra': torch.tensor(spectra, dtype=torch.float32),
        'weights': torch.tensor(weights, dtype=torch.float32),
        'theta': torch.tensor(theta, dtype=torch.float32),
        'n_components': n_components,
        'model_type': model_type,
        'n_freq': n_freq,
        'n_samples': len(spectra),
    }
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, save_path)
    logger.info(f"Saved {len(spectra)} simulations to {save_path}")


def load_simulations(load_path: Path) -> Dict[str, Any]:
    """
    Load simulations from torch file.
    
    Parameters
    ----------
    load_path : Path
        Path to .pt file
        
    Returns
    -------
    dict
        Dictionary with spectra, weights, theta, and metadata
    """
    data = torch.load(load_path, map_location='cpu', weights_only=False)
    
    # Convert to numpy for compatibility
    result = {
        'spectra': data['spectra'].numpy() if isinstance(data['spectra'], torch.Tensor) else data['spectra'],
        'weights': data['weights'].numpy() if isinstance(data['weights'], torch.Tensor) else data['weights'],
        'theta': data['theta'].numpy() if isinstance(data['theta'], torch.Tensor) else data['theta'],
        'n_components': data['n_components'],
        'model_type': data['model_type'],
        'n_freq': data['n_freq'],
        'n_samples': data.get('n_samples', len(data['spectra'])),
    }
    
    if 'metadata' in data:
        result['metadata'] = data['metadata']
    
    logger.info(f"Loaded {result['n_samples']} simulations from {load_path}")
    return result


def create_simulation_dataloader(
    spectra: np.ndarray,
    weights: np.ndarray,
    theta: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    validation_fraction: float = 0.0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create DataLoader(s) from simulation data.
    
    Parameters
    ----------
    spectra : np.ndarray
        Simulated spectra
    weights : np.ndarray
        Channel weights
    theta : np.ndarray, optional
        Parameters
    labels : np.ndarray, optional
        Class labels
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle training data
    num_workers : int
        Number of data loading workers
    validation_fraction : float
        Fraction of data to use for validation (0 for no validation set)
        
    Returns
    -------
    tuple
        (train_loader, val_loader) where val_loader is None if validation_fraction=0
    """
    n_samples = len(spectra)
    
    if validation_fraction > 0:
        n_val = int(n_samples * validation_fraction)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        # Create datasets
        train_dataset = SimulationDataset(
            spectra[train_indices],
            weights[train_indices],
            theta[train_indices] if theta is not None else None,
            labels[train_indices] if labels is not None else None,
        )
        
        val_dataset = SimulationDataset(
            spectra[val_indices],
            weights[val_indices],
            theta[val_indices] if theta is not None else None,
            labels[val_indices] if labels is not None else None,
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        return train_loader, val_loader
    
    else:
        dataset = SimulationDataset(spectra, weights, theta, labels)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        return loader, None


def prepare_classifier_data(
    simulations_dir: Path,
    min_components: int = 1,
    max_components: int = 5,
    model_types: Optional[List[str]] = None,
    cross_model_training: bool = False,
    validation_fraction: float = 0.2,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader, int, Dict[int, Tuple[str, int]]]:
    """
    Load saved simulations and prepare DataLoaders for classifier training.
    
    Parameters
    ----------
    simulations_dir : Path
        Directory containing saved simulations
    min_components : int
        Minimum number of components
    max_components : int
        Maximum number of components
    model_types : List[str], optional
        List of model types
    cross_model_training : bool
        If True, load from all model types
    validation_fraction : float
        Validation set fraction
    batch_size : int
        Batch size
        
    Returns
    -------
    tuple
        (train_loader, val_loader, n_freq, class_to_label)
    """
    if model_types is None:
        model_types = ["faraday_thin"]
    
    simulations_dir = Path(simulations_dir)
    
    all_spectra = []
    all_weights = []
    all_labels = []
    n_freq = None
    
    # Build class-to-label mapping
    class_to_label = {}
    class_idx = 0
    
    if cross_model_training:
        # Load from all model types
        for model_type in model_types:
            for n_comp in range(min_components, max_components + 1):
                # Try .pt first, then fall back to .pkl
                sim_path = simulations_dir / f"simulations_{model_type}_n{n_comp}.pt"
                if not sim_path.exists():
                    sim_path = simulations_dir / f"simulations_{model_type}_n{n_comp}.pkl"
                
                if not sim_path.exists():
                    raise FileNotFoundError(f"Simulations not found: {sim_path}")
                
                # Load based on extension
                if sim_path.suffix == '.pt':
                    sim_data = load_simulations(sim_path)
                else:
                    # Legacy pickle support
                    import pickle
                    with open(sim_path, 'rb') as f:
                        sim_data = pickle.load(f)
                
                spectra = sim_data['spectra']
                weights = sim_data['weights']
                
                if n_freq is None:
                    n_freq = weights.shape[1] if weights.ndim > 1 else len(weights)
                
                n_samples = len(spectra)
                labels = np.full(n_samples, class_idx, dtype=np.int64)
                class_to_label[class_idx] = (model_type, n_comp)
                
                all_spectra.append(spectra)
                all_weights.append(weights)
                all_labels.append(labels)
                
                logger.info(f"Loaded {n_samples} samples for {model_type} n={n_comp} (class {class_idx})")
                class_idx += 1
    else:
        # Single-model training
        model_type = model_types[0]
        for n_comp in range(min_components, max_components + 1):
            sim_path = simulations_dir / f"simulations_{model_type}_n{n_comp}.pt"
            if not sim_path.exists():
                sim_path = simulations_dir / f"simulations_{model_type}_n{n_comp}.pkl"
            
            if not sim_path.exists():
                raise FileNotFoundError(f"Simulations not found: {sim_path}")
            
            if sim_path.suffix == '.pt':
                sim_data = load_simulations(sim_path)
            else:
                import pickle
                with open(sim_path, 'rb') as f:
                    sim_data = pickle.load(f)
            
            spectra = sim_data['spectra']
            weights = sim_data['weights']
            
            if n_freq is None:
                n_freq = weights.shape[1] if weights.ndim > 1 else len(weights)
            
            n_samples = len(spectra)
            # Class index is relative to min_components
            labels = np.full(n_samples, n_comp - min_components, dtype=np.int64)
            class_to_label[n_comp - min_components] = (model_type, n_comp)
            
            all_spectra.append(spectra)
            all_weights.append(weights)
            all_labels.append(labels)
            
            logger.info(f"Loaded {n_samples} samples for {n_comp}-component model")
    
    # Combine
    all_spectra = np.vstack(all_spectra)
    all_weights = np.vstack(all_weights)
    all_labels = np.concatenate(all_labels)
    
    logger.info(f"Total: {len(all_labels)} samples")
    
    # Create DataLoaders
    train_loader, val_loader = create_simulation_dataloader(
        all_spectra,
        all_weights,
        labels=all_labels,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
    )
    
    return train_loader, val_loader, n_freq, class_to_label
