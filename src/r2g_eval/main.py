import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any

from r2g_eval.data_generators import EmbeddingDataGenerator, RandomDataGenerator, DataGenerator
from r2g_eval.experiment import ExperimentRunner
from r2g_eval.mpnn_trainer import TrainingConfig
from r2g_eval.problems import MoreThanNeighborsProblemGenerator, ProblemGenerator, SharedAttributesProblemGenerator
from r2g_eval.algorithms import get_default_algorithms
from r2g_eval.models import RDBInstance, ProblemInstance

try:
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
except ImportError:
    pass

class RelBenchTaskProblemGenerator(ProblemGenerator):
    def __init__(self, task: Any):
        super().__init__(name=getattr(task, "name", "relbench_task"), description="Extracts labels from RelBench task.")
        self.task = task
        self.target_table = getattr(task, "entity_table", "")
        self.target_col = getattr(task, "target_col", "")
        self.entity_col = getattr(task, "entity_col", "")
        
        self.labels_dict = {}
        for split in ['train', 'val', 'test']:
            try:
                table = task.get_table(split)
                if table is not None and getattr(table, "df", None) is not None:
                    df = table.df
                    if self.entity_col in df.columns and self.target_col in df.columns:
                        for _, row in df.iterrows():
                            self.labels_dict[str(row[self.entity_col])] = float(row[self.target_col])
            except Exception:
                pass

    def generate_problem_from_rdb(self, rdb_instance: RDBInstance) -> ProblemInstance:
        expected_properties = {}
        if self.target_table in rdb_instance.data:
            target_df = rdb_instance.data[self.target_table]
            if 'ID' in target_df.columns:
                for idx in target_df['ID']:
                    str_idx = str(idx)
                    if str_idx in self.labels_dict:
                        expected_properties[str_idx] = self.labels_dict[str_idx]
        
        return ProblemInstance(
            instance_id=rdb_instance.instance_id,
            task_name=self.name,
            rdb_instance=rdb_instance,
            expected_properties=expected_properties
        )


class RelBenchSimpleStackGenerator(DataGenerator):
    def __init__(self, db: Any, n_instances: int, sample_frac: float = 0.05, rng_seed: int = 42, max_value_freq: int = 100):
        super().__init__(name="relbench_simple_stack")
        self.n_instances = n_instances
        self.sample_frac = sample_frac
        self.rng_seed = rng_seed
        self.max_value_freq = max_value_freq
        self.db = db
        
    def generate(self) -> List[RDBInstance]:
        rng = np.random.default_rng(self.rng_seed)
        instances = []
        
        for i in range(self.n_instances):
            sampled_data = {}
            instance_fkeys = {}
            
            for table_name, table in self.db.table_dict.items():
                df = table.df.copy()
                
                if table.fkey_col_to_pkey_table:
                    instance_fkeys[table_name] = table.fkey_col_to_pkey_table
                
                pkey_col = table.pkey_col
                if pkey_col is not None and pkey_col in df.columns:
                    df = df.rename(columns={pkey_col: 'ID'})
                elif 'ID' not in df.columns and 'id' in df.columns:
                    df = df.rename(columns={'id': 'ID'})
                elif 'ID' not in df.columns:
                    df.insert(0, 'ID', [f"{table_name}_{j}" for j in range(len(df))])
                    
                known_fkeys = list(table.fkey_col_to_pkey_table.keys()) if table.fkey_col_to_pkey_table else []
                cols_to_keep = ['ID']
                
                for col in df.columns:
                    if col == 'ID': 
                        continue
                    if col in known_fkeys:
                        cols_to_keep.append(col)
                        continue
                    if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                        cols_to_keep.append(col)
                
                if table_name == 'posts' and 'Score' in df.columns and 'Score' not in cols_to_keep:
                     cols_to_keep.append('Score')
                if table_name == 'comments' and 'Score' in df.columns and 'Score' not in cols_to_keep:
                     cols_to_keep.append('Score')

                sample_size = max(1, int(len(df) * self.sample_frac))
                sampled_df = df[cols_to_keep].sample(n=sample_size, random_state=int(rng.integers(0, 1000000)))
                
                if table_name == 'users':
                    embeddings = np.ones((len(sampled_df), 1)).tolist()
                elif table_name == 'posts' and 'Score' in sampled_df.columns:
                    scores = pd.to_numeric(sampled_df['Score'], errors='coerce').fillna(0.0).astype(float).values
                    embeddings = scores.reshape(-1, 1).tolist()
                elif table_name == 'comments' and 'Score' in sampled_df.columns:
                    scores = pd.to_numeric(sampled_df['Score'], errors='coerce').fillna(0.0).astype(float).values
                    embeddings = scores.reshape(-1, 1).tolist()
                else:
                    embeddings = np.ones((len(sampled_df), 1)).tolist()
                
                sampled_df['Embeddings'] = embeddings
                
                for col in sampled_df.columns:
                    if col == 'ID' or col == 'Embeddings':
                        continue
                    val_counts = sampled_df[col].value_counts()
                    to_remove = val_counts[val_counts > self.max_value_freq].index
                    if len(to_remove) > 0:
                        sampled_df[col] = sampled_df[col].astype(object)
                        sampled_df.loc[sampled_df[col].isin(to_remove), col] = np.nan
                
                sampled_data[table_name] = sampled_df
                
            instances.append(
                RDBInstance(
                    instance_id=f"stack_sample_{i}",
                    task_name="user_engagement",
                    data=sampled_data,
                    fkeys=instance_fkeys
                )
            )
        return instances


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def _get_pos_weight_loss(train_problems):
    all_labels = []
    for p in train_problems:
        if p.expected_properties:
            all_labels.extend(list(p.expected_properties.values()))
            
    if not all_labels:
        return nn.BCEWithLogitsLoss()
        
    total = len(all_labels)
    positives = sum(all_labels)
    
    if positives == 0 or positives == total:
        return nn.BCEWithLogitsLoss()
        
    # For highly imbalanced graphs, use Focal Loss
    # Alpha balances positive/negative class importance
    alpha = positives / total if positives < total / 2 else 0.25
    return FocalLossWithLogits(alpha=alpha, gamma=2.0)


def main_relbench():
    cache_file = "relbench_simple_stack_cache.pkl"
    
    if os.path.exists(cache_file):
        print(f"Loading cached generated problems from '{cache_file}'...")
        with open(cache_file, "rb") as f:
            train_problems, test_problems, task_name = pickle.load(f)
        print("Data loaded successfully!")
        
    else:
        print("No cache found. Loading RelBench Stack Data and Task...")
        dataset = get_dataset("rel-stack")
        
        try:
            task = get_task("rel-stack", "user-engagement")
        except Exception:
            task = get_task("rel-stack", "rel-stack-user-engagement")
            
        db = dataset.get_db()
        
        print("Generating simple instances (No NLP)...")
        generator = RelBenchSimpleStackGenerator(db=db, n_instances=12, sample_frac=0.08, rng_seed=42, max_value_freq=100)
        instances = generator.generate()
        
        train_rdb = instances[:10]
        test_rdb = instances[10:]
        
        print("Generating problems using RelBench original labels...")
        problem_gen = RelBenchTaskProblemGenerator(task)
        
        train_problems = [problem_gen.generate_problem_from_rdb(rdb) for rdb in train_rdb]
        test_problems = [problem_gen.generate_problem_from_rdb(rdb) for rdb in test_rdb]

        train_problems = [p for p in train_problems if len(p.expected_properties) > 0]
        test_problems = [p for p in test_problems if len(p.expected_properties) > 0]
        
        if not train_problems or not test_problems:
            print("Warning: Sampling fraction was too low, no labeled entities were included in the samples.")
            return

        task_name = f"relbench_task_{problem_gen.name}"
        
        print(f"Saving generated problems to '{cache_file}'...")
        with open(cache_file, "wb") as f:
            pickle.dump((train_problems, test_problems, task_name), f)
        print("Cache saved successfully!")

    print("\nStarting Experiment...")
    loss_fn = _get_pos_weight_loss(train_problems)
    config = TrainingConfig(hidden_dim=64, num_layers=5, lr=1e-2, epochs=25, batch_size=1)
    algorithms = get_default_algorithms()
    
    experiment = ExperimentRunner(
        train_problems=train_problems,
        test_problems=test_problems,
        algorithms=algorithms,
        training_config=config,
        task_name=task_name,
        loss_fn=loss_fn
    )
    
    experiment.run()


def main():
    train_gen = EmbeddingDataGenerator(
        n_instances=1000, 
        n_rows_range=(10, 20), 
        n_cols=4, 
        n_categories=8, 
        rng_seed=42, 
        name_prefix="synthetic"
    )
    test_gen = EmbeddingDataGenerator(
        n_instances=200, 
        n_rows_range=(10, 20), 
        n_cols=4, 
        n_categories=8, 
        rng_seed=99, 
        name_prefix="synthetic"
    )

    problem_gen = SharedAttributesProblemGenerator(N=3)
    problem_gen = MoreThanNeighborsProblemGenerator(N=4)
    print("Generating data and attaching labels...")
    train_rdb = train_gen.generate()
    test_rdb = test_gen.generate()
    
    train_problems = [problem_gen.generate_problem_from_rdb(rdb) for rdb in train_rdb]
    test_problems = [problem_gen.generate_problem_from_rdb(rdb) for rdb in test_rdb]

    loss_fn = _get_pos_weight_loss(train_problems)
    config = TrainingConfig(hidden_dim=32, num_layers=3, lr=1e-2, epochs=35, batch_size=16)
    algorithms = get_default_algorithms()
    
    experiment = ExperimentRunner(
        train_problems=train_problems,
        test_problems=test_problems,
        algorithms=algorithms,
        training_config=config,
        task_name=problem_gen.name,
        loss_fn=loss_fn
    )
    
    experiment.run()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "relbench":
        main_relbench()
    else:
        main()