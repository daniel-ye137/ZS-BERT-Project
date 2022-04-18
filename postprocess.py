import numpy as np
from sklearn.decomposition import PCA
import argparse


parser = argparse.ArgumentParser(description='postprocess word embeddings')
parser.add_argument("file", help="file containing embeddings to be processed")
args = parser.parse_args()
N = 2
embedding_file = args.file
embs = []

#map indexes of word vectors in matrix to their corresponding words
idx_to_word = dict()
dimension = 0
#append each vector to a 2-D matrix and calculate average vector
with open(embedding_file, 'rb') as f:
    first_line = []
    for line in f: 
        first_line = line.rstrip().split()
        dimension = len(first_line) - 1
        if dimension < 100 :
            continue
        print("dimension: ", dimension)
        
        break
    avg_vec = [0] * dimension
    vocab_size = 0
    word = str(first_line[0].decode("utf-8"))
    word = word.split("_")[0]
    # print(word)
    idx_to_word[vocab_size] = word
    vec = [float(x) for x in first_line[1:]]
    avg_vec = [vec[i] + avg_vec[i] for i in range(len(vec))]
    vocab_size += 1
    embs.append(vec)
    for line in f:
        line = line.rstrip().split()
        word = str(line[0].decode("utf-8"))
        word = word.split("_")[0]
        idx_to_word[vocab_size] = word
        vec = [float(x) for x in line[1:]]
        avg_vec = [vec[i] + avg_vec[i] for i in range(len(vec))]
        vocab_size += 1
        embs.append(vec)
    avg_vec = [x / vocab_size for x in avg_vec]
# convert to numpy array
embs = np.array(embs)

#subtract average vector from each vector
for i in range(len(embs)):
    new_vec = [embs[i][j] - avg_vec[j] for j in range(len(avg_vec))]
    embs[i] = np.array(new_vec)

#principal component analysis using sklearn
pca = PCA()
pca.fit(embs)

#remove the top N components from each vector
for i in range(len(embs)):
    preprocess_sum = [0] * dimension
    for j in range(N):
        princip = np.array(pca.components_[j])
        preprocess = princip.dot(embs[i])
        preprocess_vec = [princip[k] * preprocess for k in range(len(princip))]
        preprocess_sum = [preprocess_sum[k] + preprocess_vec[k] for k in range(len(preprocess_sum))]
    embs[i] = np.array([embs[i][j] - preprocess_sum[j] for j in range(len(preprocess_sum))])

file = open("postprocessed_embeddings.txt", "w+", encoding="utf-8")

#write back new word vector file
idx = 0
for vec in embs:
    file.write(idx_to_word[idx])
    file.write(" ")
    for num in vec:
        file.write(str(num))
        file.write(" ")
    file.write("\n")
    idx+=1
file.close()

print("Wrote: ", len(embs), "word embeddings")