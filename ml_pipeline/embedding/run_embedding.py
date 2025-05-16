import os
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import scale
from PyWGCNA import WGCNA
from gensim.models.word2vec import Word2Vec

def parse_args():
    parser = argparse.ArgumentParser(description='Node2Vec embeddings from gene expression')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file (genes x expression).')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to write outputs.')
    parser.add_argument('--vector-size', type=int, required=True, help='Embedding dimension.')
    parser.add_argument('--walks', type=int, default=10, help='Number of walks per node.')
    parser.add_argument('--length', type=int, default=10, help='Length of each walk.')
    parser.add_argument('--batch-id', type=str, required=True, help='Batch identifier for this run.')
    parser.add_argument('--reuse-walks', action='store_true', help='If set, reuse saved walks for embedding.')
    return parser.parse_args()

def create_graphs(data, batch_id, output_dir):
    print(f"Creating graph for batch {batch_id}...")

    Genes_type = data[['Genes', 'Gene_type']]
    expression_data = data.drop(columns=['Genes', 'Gene_type'])
    Dat_exp = expression_data.T
    Dat_exp.columns = Genes_type['Genes']
    scaled_expression_data = pd.DataFrame(
        scale(Dat_exp, axis=0),
        index=Dat_exp.index,
        columns=Dat_exp.columns
    )

    # Pick soft threshold and create adjacency matrix
    print("Calculating soft threshold...")
    sft_power = WGCNA.pickSoftThreshold(scaled_expression_data)
    power_estimate = sft_power[0] 
    print(f"Power estimate: {power_estimate}")
    adjacency_matrix = WGCNA.adjacency(scaled_expression_data, power=power_estimate)
    adjacency_matrix = np.where(np.isnan(adjacency_matrix), adjacency_matrix[~np.isnan(adjacency_matrix)].mean(), adjacency_matrix)

    # Build graph
    print(f"Building graph for batch {batch_id}...")
    G = nx.Graph()
    G.add_nodes_from(Genes_type['Genes'])

    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            G.add_edge(Genes_type['Genes'][i], Genes_type['Genes'][j], weight=adjacency_matrix[i, j])

    # Save graph and metadata
    graph_file = os.path.join(output_dir, f'graph_batch_{batch_id}.pkl')
    genes_file = os.path.join(output_dir, f'genes_type_batch_{batch_id}.pkl')
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)
    with open(genes_file, 'wb') as f:
        pickle.dump(Genes_type, f)

    return G, Genes_type

def compute_probabilities(G, probs, batch_id, output_dir, p=1, q=1):
    print(f"Computing transition probabilities for batch {batch_id}...")
    with tqdm(total=G.number_of_edges(), desc=f'Computing probs for batch {batch_id}') as pbar:
        for source_node in G.nodes():
            for current_node in G.neighbors(source_node):
                probs_ = []
                for dest in G.neighbors(current_node):
                    if source_node == dest:
                        prob = G[current_node][dest].get('weight', 1) / p
                    elif dest in G.neighbors(source_node):
                        prob = G[current_node][dest].get('weight', 1)
                    else:
                        prob = G[current_node][dest].get('weight', 1) / q
                    probs_.append(prob)
                probs[source_node]['probabilities'][current_node] = np.array(probs_) / np.sum(probs_)
                pbar.update(1)

    prob_file = os.path.join(output_dir, f'probs_batch_{batch_id}.pkl')
    with open(prob_file, 'wb') as f:
        pickle.dump(probs, f)
    return probs

def generate_random_walks(G, probs, batch_id, max_walks, walk_len, output_dir):
    print(f"Generating random walks for batch {batch_id}...")
    walks = []
    for start_node in G.nodes():
        for _ in range(max_walks):
            walk = [start_node]
            walk_options = list(G[start_node])
            if len(walk_options) == 0:
                break
            first_step = np.random.choice(walk_options)
            walk.append(first_step)

            for _ in range(walk_len - 2):
                walk_options = list(G[walk[-1]])
                if len(walk_options) == 0:
                    break
                probabilities = probs[walk[-2]]['probabilities'][walk[-1]]
                next_step = np.random.choice(walk_options, p=probabilities)
                walk.append(next_step)
            walks.append(walk)

    np.random.shuffle(walks)
    walks = [list(map(str, walk)) for walk in walks]

    walks_file = os.path.join(output_dir, f'walks_batch_{batch_id}.pkl')
    with open(walks_file, 'wb') as f:
        pickle.dump(walks, f)

    return walks

def Node2Vec(generared_walks, batch_id, vector_size, window_size=20):
    print(f"Training Node2Vec model for batch {batch_id}...")
    model = Word2Vec(sentences=generared_walks, vector_size=vector_size, window=window_size)
    model_file = os.path.join(output_dir, f'node2vec_model_batch_{batch_id}.model')
    model.save(model_file)
    return model.wv

def load_saved_data(batch_id, base_dir):
    print(f"\nLoading saved data for batch {batch_id}...")
    # Load walks
    walks_file = os.path.join(base_dir, f'walks_batch_{batch_id}.pkl')
    with open(walks_file, 'rb') as f:
        walks = pickle.load(f)
    # Load genes type
    genes_type_file = os.path.join(base_dir, f'genes_type_batch_{batch_id}.pkl')
    with open(genes_type_file, 'rb') as f:
        genes_type = pickle.load(f)
    return walks, genes_type

def generate_embeddings_for_vector_sizes(batch_ids, vector_sizes, base_dir, window_size=20):
    for vector_size in vector_sizes:
        print(f"\nGenerating embeddings for vector size {vector_size}...")
        # Initialize an empty DataFrame for merged embeddings across all batches
        embeddings = pd.DataFrame()
        for batch_id in batch_ids:
            print(f"Processing batch {batch_id} for vector size {vector_size}...")
            walks, genes_type = load_saved_data(batch_id, base_dir)
            model = Word2Vec(
                sentences=walks,
                vector_size=vector_size,
                window=window_size)
            df = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)
            df = df.reset_index().rename(columns={'index': 'Genes'})
            df = pd.merge(genes_type, df, on='Genes', how='inner')
            embeddings = pd.concat([embeddings, df]).reset_index(drop=True)
        embeddings = embeddings.drop_duplicates(subset=['Genes']).reset_index(drop=True)
        embedding_file = os.path.join(base_dir, f'train_embedding_{vector_size}.csv')
        embeddings.to_csv(embedding_file, index=False)
        print(f"Final embeddings saved to '{embedding_file}'.")
    
def main():
    global args 
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = pd.read_csv(args.input)
    pos_data = data.loc[data['Gene_type'] == 1]
    neg_data = data.loc[data['Gene_type'] == 0]
    embeddings = pd.DataFrame()
    batch_id = 1
    if not args.reuse_walks:
        while len(neg_data) > 0:
            print(f"\nProcessing batch {batch_id}...")
            sample = neg_data.sample(n=min(400, len(neg_data)), replace=False)
            neg_data = neg_data.drop(sample.index).reset_index(drop=True)
            final = pd.concat([pos_data, sample]).reset_index(drop=True)
            
            # Generate graph, probabilities, walks, and embeddings
            G, Genes_type = create_graphs(final, batch_id, args.output_dir)
            probs = defaultdict(dict)
            for node in G.nodes():
                probs[node]['probabilities'] = dict()
            cp = compute_probabilities(G, probs, batch_id, args.output_dir, p=1, q=1)
            walks = generate_random_walks(G, cp, batch_id, args.walks, args.length,args.output_dir)

            # ---------- Node2Vec ----------
            n2v_emb = Node2Vec(walks, batch_id, args.vector_size, window_size=20,args.output_dir)

            # ---------- Merge embeddings ----------
            df = pd.DataFrame(n2v_emb.vectors)
            df = pd.merge(Genes_type, df, left_index=True, right_index=True)
            embeddings = pd.concat([embeddings, df]).reset_index(drop=True)

            print(f"Batch {batch_id} embeddings obtained.")
            batch_id += 1

        out_file = os.path.join(args.output_dir, f'train_embedding_{args.vector_size}.csv')
        embeddings.to_csv(out_file, index=False)
        print(f"\nFinal embeddings saved to '{out_file}'.")
    else:
        print(f"\nGenerating embeddings from saved data for batch {args.batch_id}...")
        generate_embeddings_for_vector_sizes(
            batch_ids=[args.batch_id],
            vector_sizes=[args.vector_size],
            base_dir=args.output_dir
        )
if __name__ == '__main__':
    main()
