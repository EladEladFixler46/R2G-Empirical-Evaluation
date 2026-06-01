import time
import numpy as np
from typing import List, Dict, Any

from r2g_eval.models import ProblemInstance
from r2g_eval.algorithms import BaseR2GAlgorithm
from r2g_eval.mpnn_trainer import TrainingConfig, train, evaluate, EvaluationResults

class ExperimentRunner:
    def __init__(
        self,
        train_problems: List[ProblemInstance],
        test_problems: List[ProblemInstance],
        algorithms: List[BaseR2GAlgorithm],
        training_config: TrainingConfig,
        task_name: str = "custom_task",
        loss_fn: Any = None
    ):
        self.train_problems = train_problems
        self.test_problems = test_problems
        self.algorithms = algorithms
        self.training_config = training_config
        self.task_name = task_name
        self.loss_fn = loss_fn

    def _calculate_label_stats(self, problems: List[ProblemInstance]) -> Dict[str, float]:
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

        for algo in self.algorithms:
            print(f"\n{'-' * 65}")
            print(f" Evaluating Algorithm: {algo.name}")
            print(f"{'-' * 65}")

            print(f"\n[2] Training MPNN ({self.training_config.num_layers} layers, {self.training_config.hidden_dim} hidden dim)...")
            t_start_train = time.perf_counter()
            model, history = train(
                train_problems=self.train_problems,
                test_problems=self.test_problems,
                algorithm=algo,
                config=self.training_config,
                loss_fn=self.loss_fn
            )
            t_train = time.perf_counter() - t_start_train
            print(f"    Training finished in {t_train:.2f}s")

            print(f"\n[3] Testing Model...")
            t_start_test = time.perf_counter()
            eval_res = evaluate(model, self.test_problems, threshold=0.5, loss_fn=self.loss_fn)
            t_test = time.perf_counter() - t_start_test
            print(f"    Testing finished in {t_test:.2f}s")
            print(eval_res.summary())

            experiment_data["results"][algo.name] = {
                "history": history,
                "evaluation": eval_res,
                "timings": {
                    "training": t_train,
                    "testing": t_test
                }
            }

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

        self._show_interactive_results(experiment_data)

        return experiment_data

    def _show_interactive_results(self, experiment_data: Dict[str, Any]) -> None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        import numpy as np
        from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score

        fig = plt.figure(figsize=(12, 6))
        ax_roc = fig.add_axes([0.05, 0.25, 0.45, 0.65])
        ax_metrics = fig.add_axes([0.55, 0.25, 0.4, 0.65])
        ax_metrics.axis('off')
        ax_slider = fig.add_axes([0.1, 0.1, 0.8, 0.05])

        slider = Slider(ax_slider, 'Threshold', 0.0, 1.0, valinit=0.5, valstep=0.01)

        colors = ['blue', 'orange', 'green', 'red']
        for idx, (algo_name, data) in enumerate(experiment_data["results"].items()):
            res = data["evaluation"]
            if not hasattr(res, 'y_true') or not res.y_true:
                continue
            fpr, tpr, _ = roc_curve(res.y_true, res.y_prob)
            ax_roc.plot(fpr, tpr, label=f"{algo_name} (AUC={res.auc_roc:.3f})", color=colors[idx % len(colors)])

        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, linestyle=':', alpha=0.7)

        metrics_text = ax_metrics.text(0.0, 1.0, "", verticalalignment='top', fontfamily='monospace', fontsize=11)

        def update(val):
            thresh = slider.val
            text_str = f"Metrics at Threshold: {thresh:.2f}\n"
            text_str += "="*55 + "\n"
            text_str += f"{'Algorithm':<15} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}\n"
            text_str += "-"*55 + "\n"

            for algo_name, data in experiment_data["results"].items():
                res = data["evaluation"]
                if not hasattr(res, 'y_true') or not res.y_true:
                    continue
                y_t = np.array(res.y_true)
                y_p = np.array(res.y_prob)
                y_pred = (y_p >= thresh).astype(int)

                acc = accuracy_score(y_t, y_pred)
                prec = precision_score(y_t, y_pred, zero_division=0)
                rec = recall_score(y_t, y_pred, zero_division=0)
                f1 = f1_score(y_t, y_pred, zero_division=0)

                text_str += f"{algo_name:<15} | {acc:<7.4f} | {prec:<7.4f} | {rec:<7.4f} | {f1:<7.4f}\n"

            metrics_text.set_text(text_str)
            fig.canvas.draw_idle()

        update(0.5)
        slider.on_changed(update)
        plt.show()