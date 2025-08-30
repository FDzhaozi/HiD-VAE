from FlagEmbedding import FlagModel
import torch
from torch.nn.functional import cosine_similarity

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
print(f"CUDA available: {'Yes' if use_cuda else 'No'}")

# Load model (will auto-download on first run)
# Will automatically use GPU if CUDA is available
model = FlagModel('BAAI/bge-base-zh-v1.5', 
                  query_instruction_for_retrieval="Generate representation for this sentence to retrieve relevant articles:",
                  use_fp16=True)  # Enable FP16 acceleration

# Encode sentences
sentences = ["2D culture", "Anime"]
embeddings = model.encode(sentences)

print(f"\n'2D culture' vector (first 5 dims): {embeddings[0][:5]}")
print(f"'Anime' vector (first 5 dims): {embeddings[1][:5]}")

# Convert numpy arrays to torch tensors
embedding1 = torch.from_numpy(embeddings[0])
embedding2 = torch.from_numpy(embeddings[1])

# Calculate cosine similarity
# Need to expand 1D vectors to 2D [1, D]
similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))

print(f"\nVector dimensions: {embeddings.shape[1]}")
print(f"Cosine similarity between '2D culture' and 'Anime': {similarity.item():.4f}")
