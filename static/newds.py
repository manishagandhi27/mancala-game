from docling.document import DocumentConverter
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import nltk
nltk.download('punkt')

# Initialize components
converter = DocumentConverter()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Step 1: Convert the document
with open("your_document.pdf", "rb") as f:
    document = converter.convert(f)

# Step 2: Extract plain text content (ignore tables here)
plain_text_chunks = []
for block in document.blocks:
    if block.type == "paragraph" or block.type == "heading":
        plain_text_chunks.append(block.text)

# Step 3: Convert tables to key-value text
def convert_table_to_text(table):
    table_texts = []
    headers = table.columns
    for row in table.to_list():
        row_text = " | ".join(f"{headers[i]}: {row[i]}" for i in range(len(headers)))
        table_texts.append(f"Table Row: {row_text}")
    return table_texts

table_texts = []
for table in document.tables:
    table_texts.extend(convert_table_to_text(table))

# Combine plain text and table text
all_chunks = plain_text_chunks + table_texts

# Step 4: Token-based chunking with overlap
def split_with_overlap(chunks, chunk_size=300, overlap=50):
    final_chunks = []
    for chunk in chunks:
        tokens = tokenizer.encode(chunk)
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            final_chunks.append(chunk_text)
    return final_chunks

tokenized_chunks = split_with_overlap(all_chunks)

# Step 5: Semantic chunking
semantic_chunks = []
current_chunk = [tokenized_chunks[0]]
embeddings = model.encode(tokenized_chunks, convert_to_tensor=True)

for i in range(1, len(tokenized_chunks)):
    similarity = util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item()
    if similarity > 0.7:  # Adjust threshold as needed
        current_chunk.append(tokenized_chunks[i])
    else:
        semantic_chunks.append(" ".join(current_chunk))
        current_chunk = [tokenized_chunks[i]]

if current_chunk:
    semantic_chunks.append(" ".join(current_chunk))

# Step 6: Output the final semantic chunks
print("Final Semantic Chunks:")
for idx, chunk in enumerate(semantic_chunks):
    print(f"Chunk {idx + 1}:\n{chunk}\n")
