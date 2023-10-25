import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from gensim.models import Doc2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. Load dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
Docs = newsgroups.data

# Extract 10% of the dataset
num_samples = int(0.1 * len(Docs))
Docs = Docs[:num_samples]

# 2. Document Representations

def tfidf_representation(Docs):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2, stop_words='english')
    return vectorizer.fit_transform(Docs).todense()

def bow_representation(Docs):
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    return vectorizer.fit_transform(Docs).todense()

def doc2vec_representation(Docs):
    tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(Docs)]
    model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=100)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return np.array([model.infer_vector(doc.split()) for doc in Docs])

representations = {
    'TF-IDF': tfidf_representation,
    'BoW': bow_representation,
    'Doc2Vec': doc2vec_representation
}

# Functions to build and train GNN, cluster, and visualize for each representation
def construct_graph(matrix):
    similarity_matrix = cosine_similarity(matrix)
    edges = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(i + 1, similarity_matrix.shape[0]):
            if similarity_matrix[i, j] > 0.7:
                edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(matrix, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

class GNNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=64):
        super(GNNNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.conv3 = GCNConv(embedding_dim, input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

def train_gnn(data, input_dim):
    model = GNNNet(input_dim).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in tqdm(range(100), desc="Training GNN"):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
    return model, loss.item()

def visualize_clusters(embeddings):
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=20)
    clusters = kmeans.fit_predict(embeddings)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=clusters, cmap='jet')
    plt.colorbar()
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

errors = {}

for name, func in representations.items():
    print(f"Processing with {name} representation...")
    matrix = func(Docs)
    data = construct_graph(matrix)
    model, error = train_gnn(data, matrix.shape[1])
    errors[name] = error
    with torch.no_grad():
        embeddings = model.conv2(F.relu(model.conv1(data.x, data.edge_index)), data.edge_index).cpu().numpy()
    visualize_clusters(embeddings)

# Visualize the errors with Barchart
names = list(errors.keys())
values = list(errors.values())
plt.bar(names, values)
plt.ylabel('Reconstruction Error')
plt.title('Error comparison among representations')
print('Error:',errors)
plt.show()
