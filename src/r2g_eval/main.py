from r2g_eval.data_generators import EmbeddingDataGenerator, RandomDataGenerator
from r2g_eval.experiment import ExperimentRunner
from r2g_eval.mpnn_trainer import TrainingConfig
from r2g_eval.problems import MoreThanNeighborsProblemGenerator, SharedAttributesProblemGenerator
from r2g_eval.algorithms import get_default_algorithms


def main():
    train_gen = EmbeddingDataGenerator(
        n_instances=1000, 
        n_rows_range=(10, 20), 
        n_cols=4, 
        n_categories=8, 
        rng_seed=42, 
        name_prefix="embedded_synthetic"
    )
    test_gen = EmbeddingDataGenerator(
        n_instances=200, 
        n_rows_range=(10, 20), 
        n_cols=4, 
        n_categories=8, 
        rng_seed=99, 
        name_prefix="embedded_synthetic"
    )


    # problem_gen = MoreThanNeighborsProblemGenerator(N=12)
    problem_gen = SharedAttributesProblemGenerator(N=3)
  
    config = TrainingConfig(hidden_dim=32, num_layers=3, lr=1e-2, epochs=80, batch_size=128)
    algorithms = get_default_algorithms()
    
    experiment = ExperimentRunner(
        train_data_generator=train_gen,
        test_data_generator=test_gen,
        problem_generator=problem_gen,
        algorithms=algorithms,
        training_config=config
    )
    
    experiment.run()

if __name__ == "__main__":
    main()