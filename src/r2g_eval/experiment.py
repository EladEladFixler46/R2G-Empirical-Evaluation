import time
import numpy as np
from typing import List, Dict, Any

from r2g_eval.models import ProblemInstance
from r2g_eval.problems import ProblemGenerator
from r2g_eval.algorithms import BaseR2GAlgorithm
from r2g_eval.mpnn_trainer import TrainingConfig, train, evaluate, EvaluationResults

# Assuming data_generators.py is in the same directory
from r2g_eval.data_generators import DataGenerator, RandomDataGenerator


class ExperimentRunner:
    """Manages the end-to-end evaluation pipeline for R2G algorithms."""

    def __init__(
        self,
        train_data_generator: DataGenerator,
        test_data_generator: DataGenerator,
        problem_generator: ProblemGenerator,
        algorithms: List[BaseR2GAlgorithm],
        training_config: TrainingConfig
    ):
        self.train_data_generator = train_data_generator
        self.test_data_generator = test_data_generator
        self.problem_generator = problem_generator
        self.algorithms = algorithms
        self.training_config = training_config

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
        Executes the pipeline and returns a comprehensive dictionary of results,
        including timings, training history, and evaluation metrics.
        """
        print("=" * 65)
        print("                 R2G Experiment Pipeline                 ")
        print("=" * 65)
        
        experiment_data = {
            "metadata": {
                "problem_task": self.problem_generator.name,
                "config": self.training_config
            },
            "results": {}
        }

        # 1. Generate Data
        print("\n[1] Generating Data...")
        t_start_gen = time.perf_counter()
        train_rdb = self.train_data_generator.generate()
        test_rdb = self.test_data_generator.generate()
        t_gen = time.perf_counter() - t_start_gen
        print(f"    Generated {len(train_rdb)} train and {len(test_rdb)} test instances in {t_gen:.2f}s")

        # 2. Attach Labels
        print(f"\n[2] Attaching Labels using {self.problem_generator.name}...")
        t_start_labels = time.perf_counter()
        train_probs = [self.problem_generator.generate_problem_from_rdb(rdb) for rdb in train_rdb]
        test_probs = [self.problem_generator.generate_problem_from_rdb(rdb) for rdb in test_rdb]
        t_labels = time.perf_counter() - t_start_labels

        train_stats = self._calculate_label_stats(train_probs)
        test_stats = self._calculate_label_stats(test_probs)

        print(f"    Labels attached in {t_labels:.2f}s")
        print(f"    Train labels: {train_stats['n_nodes']} nodes, positive rate: {train_stats['positive_rate']:.3f}")
        print(f"    Test labels:  {test_stats['n_nodes']} nodes, positive rate: {test_stats['positive_rate']:.3f}")

        # 3 & 4. Train and Evaluate each Algorithm
        for algo in self.algorithms:
            print(f"\n{'-' * 65}")
            print(f" Evaluating Algorithm: {algo.name}")
            print(f"{'-' * 65}")

            # 3. Training
            print(f"\n[3] Training MPNN ({self.training_config.num_layers} layers, {self.training_config.hidden_dim} hidden dim)...")
            t_start_train = time.perf_counter()
            model, history = train(
                train_problems=train_probs,
                test_problems=test_probs,
                algorithm=algo,
                config=self.training_config
            )
            t_train = time.perf_counter() - t_start_train
            print(f"    Training finished in {t_train:.2f}s")

            # 4. Testing
            print(f"\n[4] Testing Model...")
            t_start_test = time.perf_counter()
            eval_res = evaluate(model, test_probs, threshold=0.5)
            t_test = time.perf_counter() - t_start_test
            print(f"    Testing finished in {t_test:.2f}s")
            print(eval_res.summary())

            # Store all collected data for this algorithm
            experiment_data["results"][algo.name] = {
                "history": history,           # training and test loss per epoch
                "evaluation": eval_res,       # final metrics (accuracy, F1, etc.)
                "timings": {
                    "data_generation": t_gen,
                    "label_attachment": t_labels,
                    "training": t_train,
                    "testing": t_test
                }
            }

        # 5. Final Comparison
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