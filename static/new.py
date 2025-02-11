import re
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt')

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example markdown content
markdown_content = """
## Safe Harbor Statement

Notes: In this presentation we have made certain estimates and assumptions...

## Another Section

This is a new paragraph under another heading.
"""

# Step 1: Split by newlines and merge headings with their following paragraphs
lines = markdown_content.strip().split("\n")
merged_chunks = []
current_chunk = ""

for i, line in enumerate(lines):
    if re.match(r"^##", line.strip()):  # Detect a heading
        if current_chunk:  # Save the previous chunk
            merged_chunks.append(current_chunk.strip())
        current_chunk = line  # Start a new chunk with the heading
    else:
        current_chunk += " " + line  # Append paragraph to the current chunk

# Add the last chunk
if current_chunk:
    merged_chunks.append(current_chunk.strip())

# Step 2: Apply semantic chunking based on cosine similarity
semantic_chunks = []
current_chunk = [merged_chunks[0]]
embeddings = model.encode(merged_chunks, convert_to_tensor=True)

for i in range(1, len(merged_chunks)):
    similarity = util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item()
    
    if similarity > 0.7:  # Adjust threshold for semantic coherence
        current_chunk.append(merged_chunks[i])
    else:
        semantic_chunks.append(" ".join(current_chunk))
        current_chunk = [merged_chunks[i]]

# Add the last semantic chunk
if current_chunk:
    semantic_chunks.append(" ".join(current_chunk))

# Step 3: Output the final semantic chunks
print("Final Semantic Chunks:")
for idx, chunk in enumerate(semantic_chunks):
    print(f"Chunk {idx + 1}:\n{chunk}\n")
