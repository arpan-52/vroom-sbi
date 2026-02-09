"""
Inference engine for VROOM-SBI.

Handles model loading and posterior inference with memory management.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from ..config import Configuration, MemoryConfig
from ..core.result import InferenceResult, ComponentResult, ClassifierResult
from ..core.base_classes import InferenceEngineInterface
from ..simulator.prior import sort_posterior_samples, get_params_per_component

logger = logging.getLogger(__name__)


def load_posterior(model_path: Path, device: str = 'cpu') -> Tuple[Any, Dict[str, Any]]:
    """Load a trained posterior from disk."""
    model_path = Path(model_path)
    
    if model_path.suffix == '.pt':
        data = torch.load(model_path, map_location=device, weights_only=False)
        posterior = data.get('posterior_object')
        if posterior is None:
            raise ValueError(f"No posterior object found in {model_path}")
        return posterior, data
    else:
        import pickle
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['posterior'], data


def load_classifier(model_path: Path, device: str = 'cpu') -> Any:
    """Load a trained classifier."""
    from ..training.classifier_trainer import ClassifierTrainer
    
    trainer = ClassifierTrainer(n_freq=1, n_classes=1, device=device)
    trainer.load(str(model_path))
    return trainer


class InferenceEngine(InferenceEngineInterface):
    """Main inference engine for VROOM-SBI."""
    
    def __init__(
        self,
        config: Optional[Configuration] = None,
        model_dir: str = "models",
        device: str = "cuda",
    ):
        self.config = config
        self.model_dir = Path(model_dir)
        self.device = device
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            self.device = "cpu"
        
        self.posteriors: Dict[str, Any] = {}
        self.posterior_metadata: Dict[str, Dict] = {}
        self.classifier = None
        self.memory_config = config.memory if config else MemoryConfig()
        self._models_on_device: Dict[str, str] = {}
    
    def load_models(self, max_components: int = 5, model_types: Optional[List[str]] = None):
        """Load trained posterior models."""
        if model_types is None:
            model_types = self.config.physics.model_types if self.config else ["faraday_thin"]
        
        logger.info(f"Loading models from {self.model_dir}")
        
        for model_type in model_types:
            for n in range(1, max_components + 1):
                model_path = self.model_dir / f"posterior_{model_type}_n{n}.pt"
                if not model_path.exists():
                    model_path = self.model_dir / f"posterior_{model_type}_n{n}.pkl"
                
                if model_path.exists():
                    try:
                        posterior, metadata = load_posterior(model_path, self.device)
                        key = f"{model_type}_n{n}"
                        self.posteriors[key] = posterior
                        self.posterior_metadata[key] = metadata
                        self._models_on_device[key] = self.device
                        logger.info(f"  Loaded {key}")
                    except Exception as e:
                        logger.warning(f"  Failed to load {model_path}: {e}")
        
        # Load classifier
        for ext in ['.pt', '.pkl']:
            classifier_path = self.model_dir / f"classifier{ext}"
            if classifier_path.exists():
                try:
                    self.classifier = load_classifier(classifier_path, self.device)
                    logger.info("  Loaded classifier")
                    break
                except Exception as e:
                    logger.warning(f"  Failed to load classifier: {e}")
        
        logger.info(f"Loaded {len(self.posteriors)} posterior models")
    
    def get_model_for_n(self, n_components: int, model_type: str = "faraday_thin") -> Optional[Any]:
        """Get posterior for given configuration."""
        return self.posteriors.get(f"{model_type}_n{n_components}")
    
    def run_inference(
        self,
        qu_obs: np.ndarray,
        weights: Optional[np.ndarray] = None,
        n_samples: int = 10000,
        use_classifier: bool = True,
        model_type: Optional[str] = None,
    ) -> Tuple[Dict[str, InferenceResult], str]:
        """Run inference on observed spectrum."""
        start_time = datetime.now()
        qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=self.device)
        
        if use_classifier and self.classifier is not None:
            classifier_result = self._run_classifier(qu_obs, weights)
            best_n = classifier_result.predicted_n_components
            best_model_type = classifier_result.predicted_model_type or model_type or "faraday_thin"
            selected_keys = [f"{best_model_type}_n{best_n}"]
            logger.info(f"Classifier selected: {best_model_type} N={best_n}")
        else:
            selected_keys = list(self.posteriors.keys())
            if model_type:
                selected_keys = [k for k in selected_keys if k.startswith(model_type)]
        
        results = {}
        for key in selected_keys:
            if key not in self.posteriors:
                continue
            
            posterior = self.posteriors[key]
            model_type_key = key.rsplit('_n', 1)[0]
            n_components = int(key.rsplit('_n', 1)[1])
            
            logger.info(f"Running inference for {key}...")
            
            samples = posterior.sample((n_samples,), x=qu_obs_t)
            samples_np = samples.cpu().numpy()
            
            params_per_comp = get_params_per_component(model_type_key)
            samples_np = sort_posterior_samples(samples_np, n_components, params_per_comp)
            
            log_probs = posterior.log_prob(samples, x=qu_obs_t)
            log_evidence = (torch.logsumexp(log_probs, dim=0) - np.log(n_samples)).cpu().item()
            
            components = self._parse_components(samples_np, n_components, model_type_key)
            
            results[key] = InferenceResult(
                n_components=n_components,
                model_type=model_type_key,
                log_evidence=log_evidence,
                components=components,
                all_samples=samples_np,
                n_posterior_samples=n_samples,
                inference_time_seconds=(datetime.now() - start_time).total_seconds(),
            )
        
        best_key = max(results.keys(), key=lambda k: results[k].log_evidence) if results else None
        return results, best_key
    
    def _run_classifier(self, qu_obs: np.ndarray, weights: Optional[np.ndarray] = None) -> ClassifierResult:
        """Run classifier on spectrum."""
        n_freq = len(qu_obs) // 2
        if weights is None:
            weights = np.ones(n_freq)
        
        classifier_input = np.concatenate([qu_obs, weights])
        classifier_input_t = torch.tensor(classifier_input, dtype=torch.float32, device=self.device)
        n_comp, prob_dict = self.classifier.predict(classifier_input_t)
        
        return ClassifierResult(
            predicted_n_components=n_comp,
            predicted_model_type=None,
            probabilities={str(k): v for k, v in prob_dict.items()},
            confidence=max(prob_dict.values()),
        )
    
    def _parse_components(self, samples: np.ndarray, n_components: int, model_type: str) -> List[ComponentResult]:
        """Parse posterior samples into component results."""
        params_per_comp = get_params_per_component(model_type)
        components = []
        
        for i in range(n_components):
            base_idx = i * params_per_comp
            rm_samples = samples[:, base_idx]
            
            if model_type == "faraday_thin":
                amp_samples = samples[:, base_idx + 1]
                chi0_samples = samples[:, base_idx + 2]
                q_samples = amp_samples * np.cos(2 * chi0_samples)
                u_samples = amp_samples * np.sin(2 * chi0_samples)
                
                component = ComponentResult(
                    rm_mean=np.mean(rm_samples), rm_std=np.std(rm_samples),
                    q_mean=np.mean(q_samples), q_std=np.std(q_samples),
                    u_mean=np.mean(u_samples), u_std=np.std(u_samples),
                    samples=np.column_stack([rm_samples, amp_samples, chi0_samples]),
                    chi0_mean=np.mean(chi0_samples), chi0_std=np.std(chi0_samples),
                )
            else:
                second_param = samples[:, base_idx + 1]
                amp_samples = samples[:, base_idx + 2]
                chi0_samples = samples[:, base_idx + 3]
                q_samples = amp_samples * np.cos(2 * chi0_samples)
                u_samples = amp_samples * np.sin(2 * chi0_samples)
                
                component = ComponentResult(
                    rm_mean=np.mean(rm_samples), rm_std=np.std(rm_samples),
                    q_mean=np.mean(q_samples), q_std=np.std(q_samples),
                    u_mean=np.mean(u_samples), u_std=np.std(u_samples),
                    samples=np.column_stack([rm_samples, second_param, amp_samples, chi0_samples]),
                    chi0_mean=np.mean(chi0_samples), chi0_std=np.std(chi0_samples),
                )
                
                if model_type == "burn_slab":
                    component.delta_phi_mean = np.mean(second_param)
                    component.delta_phi_std = np.std(second_param)
                else:
                    component.sigma_phi_mean = np.mean(second_param)
                    component.sigma_phi_std = np.std(second_param)
            
            components.append(component)
        
        return components
    
    def infer(self, qu_obs: np.ndarray, weights: Optional[np.ndarray] = None, n_samples: int = 10000):
        """Convenience method - run inference and return best result."""
        results, best_key = self.run_inference(qu_obs, weights=weights, n_samples=n_samples)
        if best_key is None:
            raise ValueError("No models available for inference")
        return results[best_key], results
