from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import pandas as pd

from r2g_eval.models import RDBInstance

class DataGenerator(ABC):
    """Abstract base class for all RDB data generators."""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self) -> List[RDBInstance]:
        raise NotImplementedError


class RandomDataGenerator(DataGenerator):
    """Generates synthetic random RDB instances."""
    
    def __init__(
        self,
        n_instances: int,
        n_rows_range: Tuple[int, int] = (10, 30),
        n_cols: int = 4,
        n_categories: int = 5,
        rng_seed: int = 0,
        name_prefix: str = "random_synthetic"
    ):
        super().__init__(name=f"{name_prefix}_{n_instances}")
        self.n_instances = n_instances
        self.n_rows_range = n_rows_range
        self.n_cols = n_cols
        self.n_categories = n_categories
        self.rng_seed = rng_seed

    def generate(self) -> List[RDBInstance]:
        rng = np.random.default_rng(self.rng_seed)
        instances: List[RDBInstance] = []

        n_rows_arr = rng.integers(
            self.n_rows_range[0], 
            self.n_rows_range[1] + 1, 
            size=self.n_instances
        )

        for i, n_rows in enumerate(n_rows_arr):
            ids = np.char.add("row_", np.arange(n_rows).astype(str))
            attr_matrix = rng.integers(0, self.n_categories, size=(n_rows, self.n_cols))
            col_names = [f"attr_{j}" for j in range(self.n_cols)]
            
            df = pd.DataFrame(attr_matrix, columns=col_names)
            df.insert(0, "ID", ids)

            instances.append(
                RDBInstance(
                    instance_id=f"inst_{i}_seed_{self.rng_seed}",
                    task_name="synthetic_generation",
                    data={"table": df},
                )
            )

        return instances


class EmbeddingDataGenerator(RandomDataGenerator):
    """Generates synthetic RDB instances with an 'Embeddings' column."""
    
    def __init__(
        self,
        n_instances: int,
        n_rows_range: Tuple[int, int] = (10, 30),
        n_cols: int = 4,
        n_categories: int = 5,
        rng_seed: int = 0,
        name_prefix: str = "embedded_synthetic",
        emb_dim: int = 1,
    ):
        super().__init__(n_instances, n_rows_range, n_cols, n_categories, rng_seed, name_prefix)
        self.emb_dim = emb_dim

    def generate(self) -> List[RDBInstance]:
        rng = np.random.default_rng(self.rng_seed)
        instances: List[RDBInstance] = []

        n_rows_arr = rng.integers(
            self.n_rows_range[0], 
            self.n_rows_range[1] + 1, 
            size=self.n_instances
        )

        for i, n_rows in enumerate(n_rows_arr):
            ids = np.char.add("row_", np.arange(n_rows).astype(str))
            attr_matrix = rng.integers(0, self.n_categories, size=(n_rows, self.n_cols))
            
            embeddings = rng.random(size=(n_rows, self.emb_dim)).tolist()
            
            col_names = [f"attr_{j}" for j in range(self.n_cols)]
            df = pd.DataFrame(attr_matrix, columns=col_names)
            df.insert(0, "ID", ids)
            df['Embeddings'] = embeddings

            instances.append(
                RDBInstance(
                    instance_id=f"inst_{i}_seed_{self.rng_seed}",
                    task_name="embedded_generation",
                    data={"table": df},
                )
            )

        return instances