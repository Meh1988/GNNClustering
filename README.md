# GNNClustering
Document Representation and Clustering

A Python project that represents documents using various techniques (TF-IDF, BoW, and Doc2Vec), constructs a graph, trains a Graph Neural Network (GNN) on these representations, and visualizes the resulting embeddings using t-SNE and clustering.



Features
-Document representations using:
-Term Frequency-Inverse Document Frequency (TF-IDF)
-Bag of Words (BoW)
-Doc2Vec
-Graph construction based on cosine similarity.
-Training of a Graph Neural Network (GNN) for representation.
-Visualization using t-SNE and KMeans clustering.
-Comparison of reconstruction errors for various representations.

Requirements
- Python 3.x
- NumPy
- scikit-learn
- Gensim
- Torch
- Torch-Geometric
- Matplotlib
- tqdm



Results
Upon executing the script, the system will:
- Load the 20 Newsgroups dataset.
- Extract a subset of the dataset.
- Represent documents using TF-IDF, BoW, and Doc2Vec.
- Construct a graph using cosine similarity.
- Train a Graph Neural Network on the representations.
- Visualize the resulting embeddings using t-SNE and clustering.
- Compare and visualize the reconstruction errors for each representation using a bar chart.
