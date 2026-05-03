import time
import numpy as np
from typing import List, Dict, Any

from r2g_eval.models import ProblemInstance
from r2g_eval.algorithms import BaseR2GAlgorithm
from r2g_eval.mpnn_trainer import TrainingConfig, train, evaluate, EvaluationResults

class ExperimentRunner:
    """Manages the end-to-end evaluation pipeline for R2G algorithms using pre-generated data."""

    def __init__(
        self,
        train_problems: List[ProblemInstance],
        test_problems: List[ProblemInstance],
        algorithms: List[BaseR2GAlgorithm],
        training_config: TrainingConfig,
        task_name: str = "custom_task"
    ):
        self.train_problems = train_problems
        self.test_problems = test_problems
        self.algorithms = algorithms
        self.training_config = training_config
        self.task_name = task_name

    def _calculate_label_stats(self, problems: List[ProblemInstance]) -> Dict[str, float]:
        """Calculates positive rate and node count across all problem instances."""
        all_labels = np.concatenate(
            [np.array(list(p.expected_properties.values())) for p in problems if p.expected_properties]
        )
        if len(all_labels) == 0:
            return {"n_nodes": 0, "positive_rate": 0.0}
            
        return {
            "n_nodes": int(len(all_labels)),
            "positive_rate": float(all_labels.mean())
        }

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Executes the pipeline on pre-generated problems and returns a comprehensive 
        dictionary of results, including timings, training history, and evaluation metrics.
        """
        print("=" * 65)
        print("                 R2G Experiment Pipeline                 ")
        print("=" * 65)
        
        experiment_data = {
            "metadata": {
                "problem_task": self.task_name,
                "config": self.training_config
            },
            "results": {}
        }

        train_stats = self._calculate_label_stats(self.train_problems)
        test_stats = self._calculate_label_stats(self.test_problems)

        print("\n[1] Data Statistics:")
        print(f"    Train labels: {train_stats['n_nodes']} nodes, positive rate: {train_stats['positive_rate']:.3f}")
        print(f"    Test labels:  {test_stats['n_nodes']} nodes, positive rate: {test_stats['positive_rate']:.3f}")

        # 2 & 3. Train and Evaluate each Algorithm
        for algo in self.algorithms:
            print(f"\n{'-' * 65}")
            print(f" Evaluating Algorithm: {algo.name}")
            print(f"{'-' * 65}")

            # 2. Training
            print(f"\n[2] Training MPNN ({self.training_config.num_layers} layers, {self.training_config.hidden_dim} hidden dim)...")
            t_start_train = time.perf_counter()
            model, history = train(
                train_problems=self.train_problems,
                test_problems=self.test_problems,
                algorithm=algo,
                config=self.training_config
            )
            t_train = time.perf_counter() - t_start_train
            print(f"    Training finished in {t_train:.2f}s")

            # 3. Testing
            print(f"\n[3] Testing Model...")
            t_start_test = time.perf_counter()
            eval_res = evaluate(model, self.test_problems, threshold=0.5)
            t_test = time.perf_counter() - t_start_test
            print(f"    Testing finished in {t_test:.2f}s")
            print(eval_res.summary())

            # Store all collected data for this algorithm
            experiment_data["results"][algo.name] = {
                "history": history,           # training and test loss per epoch
                "evaluation": eval_res,       # final metrics (accuracy, F1, etc.)
                "timings": {
                    "training": t_train,
                    "testing": t_test
                }
            }

        # 4. Final Comparison
        print("\n" + "=" * 65)
        print("                       Final Comparison")
        print("=" * 65)
        print(f"{'Algorithm':<18} | {'Accuracy':<10} | {'F1 Score':<10} | {'Train Time':<10}")
        print("-" * 65)
        for algo_name, data in experiment_data["results"].items():
            res = data["evaluation"]
            t_tr = data["timings"]["training"]
            print(f"{algo_name:<18} | {res.accuracy:<10.4f} | {res.f1:<10.4f} | {t_tr:<10.2f}s")
        print("=" * 65 + "\n")

        return experiment_data