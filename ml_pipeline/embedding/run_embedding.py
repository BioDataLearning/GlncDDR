
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from gensim.models import Word2Vec
from PyWGCNA import WGCNA

def create_graphs(data):
    Genes_type = data.iloc[:, :2]
    expression_data = data.drop(columns=['Genes', 'Gene_type']).T
    expression_data.columns = Genes_type['Genes']
    scaled = pd.DataFrame(scale(expression_data))
    adjacency_matrix = WGCNA.adjacency(scaled, power=5)
    adjacency_matrix = np.nan_to_num(adjacency_matrix)
    G = nx.Graph()
    G.add_nodes_from(Genes_type['Genes'])
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            G.add_edge(Genes_type['Genes'][i], Genes_type['Genes'][j], weight=adjacency_matrix[i][j])
    return G, Genes_type

def compute_probabilities(G, probs, p, q):
    for src in G.nodes():
        for cur in G.neighbors(src):
            probs_ = []
            for dest in G.neighbors(cur):
                if src == dest:
                    prob = G[cur][dest].get('weight', 1) * (1/p)
                elif dest in G.neighbors(src):
                    prob = G[cur][dest].get('weight', 1)
                else:
                    prob = G[cur][dest].get('weight', 1) * (1/q)
                probs_.append(prob)
            norm_probs = probs_ / np.sum(probs_)
            probs[src]['probabilities'][cur] = norm_probs
    return probs

def generate_random_walks(G, probs, max_walks, walk_len):
    walks = []
    for node in G.nodes():
        for _ in range(max_walks):
            walk = [node]
            neighbors = list(G[node])
            if not neighbors:
                break
            first = np.random.choice(neighbors)
            walk.append(first)
            for _ in range(walk_len - 2):
                neighbors = list(G[walk[-1]])
                if not neighbors:
                    break
                next_node = np.random.choice(neighbors, p=probs[walk[-2]]['probabilities'][walk[-1]])
                walk.append(next_node)
            walks.append(walk)
    np.random.shuffle(walks)
    return [list(map(str, walk)) for walk in walks]

def Node2Vec(walks, window_size, embedding_size):
    model = Word2Vec(sentences=walks, window=window_size, vector_size=embedding_size)
    return model.wv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--walks', type=int, default=5)
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--dim', type=int, default=100)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    G, Genes_type = create_graphs(data)
    probs = defaultdict(dict)
    for node in G.nodes():
        probs[node]['probabilities'] = dict()
    probs = compute_probabilities(G, probs, 1, 1)
    walks = generate_random_walks(G, probs, args.walks, args.length)
    embeddings = Node2Vec(walks, window_size=10, embedding_size=args.dim)
    df = pd.DataFrame(embeddings.vectors)
    df = pd.merge(Genes_type, df, left_index=True, right_index=True)
    df.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
