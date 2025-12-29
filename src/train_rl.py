"""RL-based training using PPO algorithm for YOLOv11 fine-tuning."""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

import torch
import numpy as np
from ultralytics import YOLO

from config import YOLO_CONFIG, TRAIN_CONFIG, PATHS
from reward import RewardCalculator


class YOLOHyperparameterRL:
    """Reinforcement learning agent for optimizing YOLO hyperparameters."""

    def __init__(
        self,
        model_name: str = YOLO_CONFIG["model"],
        learning_rate_bounds: Tuple[float, float] = (1e-5, 1e-3),
        momentum_bounds: Tuple[float, float] = (0.8, 0.99),
    ):
        """Initialize RL agent for YOLO hyperparameter optimization.
        
        Args:
            model_name: YOLO model to use
            learning_rate_bounds: Range for learning rate
            momentum_bounds: Range for momentum parameter
        """
        self.model_name = model_name
        self.lr_bounds = learning_rate_bounds
        self.momentum_bounds = momentum_bounds
        
        self.reward_history: List[float] = []
        self.hyperparameter_history: List[Dict[str, float]] = []
        self.best_reward = -float("inf")
        self.best_hyperparameters = None
        
    def sample_hyperparameters(self) -> Dict[str, float]:
        """Sample hyperparameters using exploration strategy."""
        lr = np.random.uniform(self.lr_bounds[0], self.lr_bounds[1])
        momentum = np.random.uniform(self.momentum_bounds[0], self.momentum_bounds[1])
        
        return {
            "lr0": float(lr),
            "momentum": float(momentum),
            "augment": np.random.choice([True, False]),
        }

    def train_with_hyperparameters(
        self,
        data_path: str,
        hyperparams: Dict[str, float],
        epochs: int = 10,
        batch_size: int = 16,
    ) -> Tuple[float, Dict[str, Any]]:
        """Train YOLO with given hyperparameters and return reward.
        
        Args:
            data_path: Path to dataset
            hyperparams: Hyperparameters to use
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            Reward score and training results
        """
        model = YOLO(f"{self.model_name}.pt")
        
        # Train with hyperparameters
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            lr0=hyperparams.get("lr0", 0.01),
            momentum=hyperparams.get("momentum", 0.937),
            device=YOLO_CONFIG["device"],
            patience=3,
            save=False,
            verbose=False,
        )
        
        # Calculate reward (higher is better)
        metrics = results.box.map50 if hasattr(results.box, 'map50') else 0.0
        precision = results.box.p[0] if hasattr(results.box, 'p') else 0.0
        recall = results.box.r[0] if hasattr(results.box, 'r') else 0.0
        
        # Combined reward: MAP50 as primary, precision and recall as secondary
        reward = 0.6 * metrics + 0.2 * precision + 0.2 * recall
        
        return float(reward), {
            "map50": float(metrics),
            "precision": float(precision),
            "recall": float(recall),
        }

    def ppo_optimization_step(
        self,
        data_path: str,
        current_hyperparams: Dict[str, float],
        baseline_reward: float = 0.0,
    ) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        """One step of PPO-style optimization.
        
        Args:
            data_path: Path to training data
            current_hyperparams: Current hyperparameters
            baseline_reward: Baseline for advantage calculation
        
        Returns:
            Updated hyperparameters, reward, and metrics
        """
        # Sample new hyperparameters
        new_hyperparams = self.sample_hyperparameters()
        
        # Train with new hyperparameters
        reward, metrics = self.train_with_hyperparameters(
            data_path, new_hyperparams
        )
        
        # PPO-style advantage calculation
        advantage = reward - baseline_reward
        
        # Accept new hyperparameters with probability based on advantage
        # This simulates PPO's clip mechanism
        accept_prob = min(1.0, np.exp(min(advantage, 1.0)))
        
        if advantage > 0 or np.random.random() < accept_prob:
            accepted_hyperparams = new_hyperparams
            self.best_reward = max(self.best_reward, reward)
            self.best_hyperparameters = accepted_hyperparams
        else:
            accepted_hyperparams = current_hyperparams
            reward = baseline_reward
        
        self.reward_history.append(reward)
        self.hyperparameter_history.append(accepted_hyperparams)
        
        return accepted_hyperparams, reward, metrics

    def optimize(
        self,
        data_path: str,
        num_iterations: int = 10,
        output_dir: str = None,
    ) -> Dict[str, Any]:
        """Run PPO optimization loop.
        
        Args:
            data_path: Path to training data
            num_iterations: Number of optimization iterations
            output_dir: Directory to save results
        
        Returns:
            Optimization results
        """
        output_dir = output_dir or PATHS["results_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        current_hyperparams = self.sample_hyperparameters()
        baseline_reward = 0.0
        
        print(f"Starting PPO optimization for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            hyperparams, reward, metrics = self.ppo_optimization_step(
                data_path, current_hyperparams, baseline_reward
            )
            
            current_hyperparams = hyperparams
            baseline_reward = reward
            
            print(f"Reward: {reward:.4f}, MAP50: {metrics['map50']:.4f}")
            
            # Save checkpoint
            if (iteration + 1) % 5 == 0:
                self._save_checkpoint(output_dir, iteration + 1)
        
        # Save final results
        results = {
            "best_reward": float(self.best_reward),
            "best_hyperparameters": self.best_hyperparameters,
            "reward_history": self.reward_history,
            "hyperparameter_history": self.hyperparameter_history,
        }
        
        with open(os.path.join(output_dir, "rl_optimization_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nOptimization complete!")
        print(f"Best reward: {self.best_reward:.4f}")
        print(f"Best hyperparameters: {self.best_hyperparameters}")
        
        return results

    def _save_checkpoint(self, output_dir: str, iteration: int):
        """Save optimization checkpoint."""
        checkpoint = {
            "iteration": iteration,
            "best_reward": self.best_reward,
            "best_hyperparameters": self.best_hyperparameters,
            "reward_history": self.reward_history,
        }
        
        checkpoint_path = os.path.join(
            output_dir, f"rl_checkpoint_iter{iteration:03d}.json"
        )
        
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)


def main():
    """Main entry point for RL training."""
    parser = argparse.ArgumentParser(
        description="RL-based YOLOv11 fine-tuning with PPO"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset.yaml",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of RL optimization iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PATHS["results_dir"]),
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=YOLO_CONFIG["model"],
        help="YOLO model to use",
    )
    
    args = parser.parse_args()
    
    # Initialize RL agent
    agent = YOLOHyperparameterRL(model_name=args.model)
    
    # Run optimization
    results = agent.optimize(
        data_path=args.data,
        num_iterations=args.iterations,
        output_dir=args.output,
    )
    
    return results


if __name__ == "__main__":
    main()
