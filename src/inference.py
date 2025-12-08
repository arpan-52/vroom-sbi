"""
Inference and model selection for RM synthesis
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import torch


@dataclass
class ComponentResult:
    """
    Results for a single RM component.
    
    Attributes
    ----------
    rm_mean : float
        Mean RM value
    rm_std : float
        Standard deviation of RM
    q_mean : float
        Mean q amplitude
    q_std : float
        Standard deviation of q amplitude
    u_mean : float
        Mean u amplitude
    u_std : float
        Standard deviation of u amplitude
    samples : np.ndarray
        Posterior samples for this component [rm, q, u]
    """
    rm_mean: float
    rm_std: float
    q_mean: float
    q_std: float
    u_mean: float
    u_std: float
    samples: np.ndarray = field(repr=False)


@dataclass
class InferenceResult:
    """
    Complete inference results.
    
    Attributes
    ----------
    n_components : int
        Number of components
    log_evidence : float
        Log marginal likelihood (evidence)
    components : List[ComponentResult]
        Results for each component
    noise_mean : float
        Mean noise level
    noise_std : float
        Standard deviation of noise level
    all_samples : np.ndarray
        All posterior samples
    """
    n_components: int
    log_evidence: float
    components: List[ComponentResult]
    noise_mean: float
    noise_std: float
    all_samples: np.ndarray = field(repr=False)


class RMInference:
    """
    RM inference with model selection.
    
    Loads trained posterior models and performs inference on observed data,
    selecting the best number of components via log evidence.
    """
    
    def __init__(self, model_dir="models", device="cuda", use_decision_layer=True):
        """
        Initialize the inference engine.
        
        Parameters
        ----------
        model_dir : str
            Directory containing trained models
        device : str
            Device to use ('cuda' or 'cpu')
        use_decision_layer : bool
            Whether to use decision layer for model selection
        """
        self.model_dir = model_dir
        self.device = device
        self.posteriors = {}
        self.use_decision_layer = use_decision_layer
        self.decision_layer = None
        
    def load_models(self, max_components=5):
        """
        Load trained posterior models and decision layer.
        
        Parameters
        ----------
        max_components : int
            Maximum number of components to load
        """
        print(f"Loading models from {self.model_dir}...")
        
        # Load worker models (posteriors)
        for n in range(1, max_components + 1):
            model_path = os.path.join(self.model_dir, f"posterior_n{n}.pkl")
            
            if os.path.exists(model_path):
                # Note: weights_only=False is required for loading sbi posterior objects
                # Only load models from trusted sources
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.posteriors[n] = model_data['posterior']
                print(f"  Loaded worker model for {n} component(s)")
            else:
                print(f"  Warning: Worker model for {n} component(s) not found at {model_path}")
        
        # Load decision layer if enabled
        if self.use_decision_layer:
            decision_path = os.path.join(self.model_dir, "decision_layer.pkl")
            if os.path.exists(decision_path):
                from .decision import DecisionLayerTrainer
                checkpoint = torch.load(decision_path, map_location=self.device)
                self.decision_layer = DecisionLayerTrainer(
                    n_freq=checkpoint['n_freq'], 
                    device=self.device
                )
                self.decision_layer.load(decision_path)
                print(f"  Loaded decision layer from {decision_path}")
            else:
                print(f"  Warning: Decision layer not found at {decision_path}")
                print(f"  Falling back to log evidence for model selection")
                self.use_decision_layer = False
        
        print(f"Loaded {len(self.posteriors)} worker models\n")
    
    def run_inference(self, qu_obs, weights=None, n_samples=10000):
        """
        Run inference for all models and select the best one.
        
        Parameters
        ----------
        qu_obs : np.ndarray or torch.Tensor
            Observed Q and U values [Q_1, ..., Q_M, U_1, ..., U_M]
        weights : np.ndarray, optional
            Channel weights. If None, assumes all weights = 1.0
        n_samples : int
            Number of posterior samples to draw
            
        Returns
        -------
        results : Dict[int, InferenceResult]
            Results for each model
        best_n : int
            Best number of components
        """
        # Convert to torch tensor
        if isinstance(qu_obs, np.ndarray):
            qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=self.device)
        else:
            qu_obs_t = qu_obs
        
        # Use decision layer if available
        if self.use_decision_layer and self.decision_layer is not None:
            print("\n" + "="*60)
            print("Using Decision Layer for Model Selection")
            print("="*60)
            
            # Prepare input for decision layer: [Q, U, weights]
            if weights is None:
                n_freq = len(qu_obs) // 2
                weights = np.ones(n_freq)
            
            # Convert qu_obs to numpy if needed
            qu_obs_np = qu_obs if isinstance(qu_obs, np.ndarray) else qu_obs.cpu().numpy()
            decision_input = np.concatenate([qu_obs_np, weights])
            decision_input_t = torch.tensor(decision_input, dtype=torch.float32, device=self.device)
            
            # Get prediction
            best_n = self.decision_layer.predict(decision_input_t)[0]
            probs = self.decision_layer.predict_proba(decision_input_t)[0]
            
            print(f"Decision layer prediction:")
            print(f"  1-component probability: {probs[0]:.3f}")
            print(f"  2-component probability: {probs[1]:.3f}")
            print(f"  Selected: {best_n} component(s)")
            print("="*60 + "\n")
            
            # Only run inference for selected model
            selected_components = [best_n]
        else:
            # Run inference for all available models
            selected_components = list(self.posteriors.keys())
        
        results = {}
        
        print("Running inference for selected model(s)...")
        
        for n_components in selected_components:
            if n_components not in self.posteriors:
                print(f"  Warning: Model for {n_components} component(s) not loaded")
                continue
            
            posterior = self.posteriors[n_components]
            print(f"\n  Model with {n_components} component(s):")
            
            # Sample from posterior
            samples = posterior.sample((n_samples,), x=qu_obs_t)
            samples_np = samples.cpu().numpy()
            
            # Compute log probability for evidence estimation
            log_probs = posterior.log_prob(samples, x=qu_obs_t)
            log_evidence = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
            log_evidence = log_evidence.cpu().item()
            
            print(f"    Log evidence: {log_evidence:.2f}")
            
            # Parse samples into components
            components = []
            for i in range(n_components):
                rm_samples = samples_np[:, i]
                q_samples = samples_np[:, n_components + i]
                u_samples = samples_np[:, 2 * n_components + i]
                
                component = ComponentResult(
                    rm_mean=np.mean(rm_samples),
                    rm_std=np.std(rm_samples),
                    q_mean=np.mean(q_samples),
                    q_std=np.std(q_samples),
                    u_mean=np.mean(u_samples),
                    u_std=np.std(u_samples),
                    samples=np.column_stack([rm_samples, q_samples, u_samples])
                )
                components.append(component)
                
                print(f"    Component {i+1}: RM = {component.rm_mean:.1f} ± {component.rm_std:.1f} rad/m²")
            
            # Noise
            noise_samples = samples_np[:, 3 * n_components]
            noise_mean = np.mean(noise_samples)
            noise_std = np.std(noise_samples)
            
            print(f"    Noise: {noise_mean:.4f} ± {noise_std:.4f}")
            
            # Store results
            result = InferenceResult(
                n_components=n_components,
                log_evidence=log_evidence,
                components=components,
                noise_mean=noise_mean,
                noise_std=noise_std,
                all_samples=samples_np
            )
            results[n_components] = result
        
        # Select best model (if not already selected by decision layer)
        if not self.use_decision_layer or self.decision_layer is None:
            best_n = max(results.keys(), key=lambda n: results[n].log_evidence)
            best_log_evidence = results[best_n].log_evidence
            
            print(f"\n{'='*60}")
            print(f"BEST MODEL (by log evidence): {best_n} component(s)")
            print(f"Log evidence: {best_log_evidence:.2f}")
            print(f"{'='*60}\n")
        
        return results, best_n
    
    def infer(self, qu_obs, weights=None, n_samples=10000):
        """
        Convenience method to run inference and return best result.
        
        Parameters
        ----------
        qu_obs : np.ndarray or torch.Tensor
            Observed Q and U values
        weights : np.ndarray, optional
            Channel weights
        n_samples : int
            Number of posterior samples
            
        Returns
        -------
        result : InferenceResult
            Best inference result
        all_results : Dict[int, InferenceResult]
            Results for all models
        """
        results, best_n = self.run_inference(qu_obs, weights=weights, n_samples=n_samples)
        return results[best_n], results
