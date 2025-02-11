from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt')

# Initialize the model and tokenizer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Example markdown content
markdown_content = """
## Safe Harbor Statement

Notes: In this presentation we have made certain estimates and assumptions...

## Another Section

This is a new paragraph under another heading.
"""

# Step 1: Split and merge headings with paragraphs (as before)
lines = [line.strip() for line in markdown_content.split("\n") if line.strip()]

merged_chunks = []
current_chunk = ""

for line in lines:
    if re.match(r"^##", line):
        if current_chunk:
            merged_chunks.append(current_chunk.strip())
        current_chunk = line
    else:
        current_chunk += " " + line

if current_chunk:
    merged_chunks.append(current_chunk.strip())

# Step 2: Tokenize and enforce chunk size and overlap
def split_with_overlap(chunks, chunk_size=300, overlap=50):
    final_chunks = []
    for chunk in chunks:
        tokens = tokenizer.encode(chunk)
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            final_chunks.append(chunk_text)
    return final_chunks

# Apply token-based chunking with overlap
tokenized_chunks = split_with_overlap(merged_chunks)

# Step 3: Apply semantic chunking on tokenized chunks
semantic_chunks = []
current_chunk = [tokenized_chunks[0]]
embeddings = model.encode(tokenized_chunks, convert_to_tensor=True)

for i in range(1, len(tokenized_chunks)):
    similarity = util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item()
    
    if similarity > 0.7:  # Adjust threshold
        current_chunk.append(tokenized_chunks[i])
    else:
        semantic_chunks.append(" ".join(current_chunk))
        current_chunk = [tokenized_chunks[i]]

if current_chunk:
    semantic_chunks.append(" ".join(current_chunk))

# Output final semantic chunks with size and overlap enforced
for idx, chunk in enumerate(semantic_chunks):
    print(f"Chunk {idx + 1}:\n{chunk}\n")
