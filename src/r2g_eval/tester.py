import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from models import RDBInstance
from algorithms import get_default_algorithms


def create_sample_databases():
    df_paper_example = pd.DataFrame({
        'ID': ['t1', 't2', 't3'],
        'A1': [1, 1, 1],
        'A2': [2, 2, 3]
    })
    
    db_paper = RDBInstance(
        instance_id="paper_db",
        task_name="test_expressiveness",
        data={"table_1": df_paper_example}
    )

    df_users = pd.DataFrame({
        'ID': ['u1', 'u2', 'u3', 'u4'],
        'Department': ['HR', 'Engineering', 'Engineering', 'HR'],
        'City': ['Tel Aviv', 'Haifa', 'Tel Aviv', 'Jerusalem']
    })

    db_company = RDBInstance(
        instance_id="company_db",
        task_name="test_company",
        data={"employees": df_users}
    )

    return [db_paper, db_company]


def visualize_graph(graph_instance, title):
    G = nx.Graph()
    
    for node_idx, label in graph_instance.node_to_id.items():
        G.add_node(node_idx, label=label)
    
    if graph_instance.edge_index.numel() > 0:
        edges_src = graph_instance.edge_index[0].tolist()
        edges_dst = graph_instance.edge_index[1].tolist()
        for u, v in zip(edges_src, edges_dst):
            G.add_edge(u, v)

    plt.figure(figsize=(10, 8))
    
    if "indirect" in title.lower():
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        node_color = ['lightgreen' if "=" in str(data.get('label', '')) else 'lightblue' 
                      for node, data in G.nodes(data=True)]
    else:
        pos = nx.circular_layout(G)
        node_color = 'lightblue'

    labels = nx.get_node_attributes(G, 'label')
    
    nx.draw(
        G, pos,
        labels=labels,
        with_labels=True,
        node_color=node_color,
        node_size=3000,
        font_size=10,
        font_weight='bold',
        edge_color='gray',
        width=2.0
    )
    
    plt.title(title, fontsize=16)
    plt.margins(0.2)
    plt.show()


def main():
    databases = create_sample_databases()
    algorithms = get_default_algorithms()

    for db in databases:
        print(f"==========================================")
        print(f"Testing Database: {db.instance_id}")
        print(f"==========================================")
        
        for algo in algorithms:
            print(f"Running Algorithm: {algo.name}...")
            
            graph = algo._run(db)
            
            num_nodes = graph.embeddings.shape[0]
            num_edges = graph.edge_index.shape[1] // 2 
            
            print(f"  -> Number of nodes: {num_nodes}")
            print(f"  -> Number of undirected edges: {num_edges}")
            print(f"  -> Metadata: {graph.metadata}")
            print(f"  -> Displaying plot...\n")
            
            title = f"Graph output for '{db.instance_id}' using '{algo.name}'"
            visualize_graph(graph, title)


if __name__ == "__main__":
    main()