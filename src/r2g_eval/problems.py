from abc import ABC
import pandas as pd
import numpy as np

from r2g_eval.models import ProblemInstance, RDBInstance


class ProblemGenerator(ABC):
    """Generates problem instances for R2G evaluation."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def generate_problem_from_rdb(self, rdb_instance: RDBInstance) -> ProblemInstance:
        """
        Subclasses must implement this method to generate problem instances from RDB data.
        """
        raise NotImplementedError


class MoreThanNeighborsProblemGenerator(ProblemGenerator):
    """Generates a problem instance where the task is to identify rows with more than N other rows that are connected."""
    def __init__(self, N: int):
        super().__init__(
            name=f"more_than_{N}_neighbors", 
            description=f"Identify rows with more than {N} other rows that are connected."
        )
        self.N = N

    def generate_problem_from_rdb(self, rdb_instance: RDBInstance) -> ProblemInstance:
        expected_properties = {}

        for df_name, df in rdb_instance.data.items():
            if df.empty:
                continue
                
            # 1. Melt: Converts the table to (ID, attribute, value) format.
            # EXCLUDE the 'Embeddings' column from being melted alongside regular attributes
            cols_to_use = [c for c in df.columns if c not in ['ID', 'Embeddings']]
            melted = df.melt(id_vars=['ID'], value_vars=cols_to_use, var_name='attr', value_name='val')
            melted = melted.dropna(subset=['val'])

            # 2. Vectorized Join: Joins the table to itself based on identical values in the same column.
            # This finds all IDs sharing at least one value.
            merged = pd.merge(melted, melted, on=['attr', 'val'])

            # 3. Self-cleaning: Removes rows where ID matched itself.
            connections = merged[merged['ID_x'] != merged['ID_y']]

            # 4. Remove duplicates: If two rows share more than one column, count as one connection.
            unique_connections = connections[['ID_x', 'ID_y']].drop_duplicates()

            # 5. Numpy Counting: Counts how many times each ID_x appears (number of neighbors).
            ids, counts = np.unique(unique_connections['ID_x'].values, return_counts=True)
            
            # Fast dictionary updates using numpy boolean arrays
            expected_properties.update(dict(zip(df['ID'].astype(str), np.zeros(len(df)))))
            expected_properties.update(dict(zip(ids.astype(str), (counts > self.N).astype(float))))
        
        return ProblemInstance(
            instance_id=rdb_instance.instance_id,
            task_name=self.name,
            rdb_instance=rdb_instance,
            expected_properties=expected_properties
        )
    
class SharedAttributesProblemGenerator(ProblemGenerator):
    """Generates a problem instance where the task is to identify rows with more than N shared attribute-value pairs."""
    def __init__(self, N: int):
        super().__init__(
            name=f"more_than_{N}_shared_attributes", 
            description=f"Identify rows with more than {N} distinct attribute-value pairs shared with other rows."
        )
        self.N = N

    def generate_problem_from_rdb(self, rdb_instance: RDBInstance) -> ProblemInstance:
        expected_properties = {}

        for df_name, df in rdb_instance.data.items():
            if df.empty:
                continue
                
            # 1. Melt the dataframe to (ID, attr, val) format
            # EXCLUDE the 'Embeddings' column from being melted alongside regular attributes
            cols_to_use = [c for c in df.columns if c not in ['ID', 'Embeddings']]
            melted = df.melt(id_vars=['ID'], value_vars=cols_to_use, var_name='attr', value_name='val')
            melted = melted.dropna(subset=['val'])

            # 2. Count distinct IDs for each (attr, val) pair
            pair_counts = melted.groupby(['attr', 'val'])['ID'].nunique().reset_index()
            
            # 3. Filter pairs that are shared by at least 2 distinct IDs
            shared_pairs = pair_counts[pair_counts['ID'] > 1]

            # 4. Keep only the rows in the melted dataframe that have these shared pairs
            shared_melted = pd.merge(melted, shared_pairs[['attr', 'val']], on=['attr', 'val'])

            # 5. Count how many shared attribute-value pairs each ID has
            id_counts = shared_melted.groupby('ID').size().reset_index(name='count')
            
            # Numpy vectorization for label condition
            expected_properties.update(dict(zip(df['ID'].astype(str), np.zeros(len(df)))))
            labels = (id_counts['count'].values > self.N).astype(float)
            expected_properties.update(dict(zip(id_counts['ID'].astype(str), labels)))
        
        return ProblemInstance(
            instance_id=rdb_instance.instance_id,
            task_name=self.name,
            rdb_instance=rdb_instance,
            expected_properties=expected_properties
        )