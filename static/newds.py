from docling.document import DocumentConverter
from docling.datamodel.document import TextItem, Table
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import re
import nltk
nltk.download('punkt')

# Initialize components
converter = DocumentConverter()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Step 1: Convert the document
with open("your_document.pdf", "rb") as f:
    document = converter.convert(f)

# Step 2: Extract Text Items and Tables
text_items = [item for item in document.items if isinstance(item, TextItem)]
tables = [table for table in document.items if isinstance(table, Table)]

# Convert TextItems into plain text chunks
text_chunks = [item.text for item in text_items]

# Step 3: Convert Tables into Searchable Text Chunks
def convert_table_to_chunks(table):
    table_chunks = []
    table_caption = f"Table Caption: {table.caption}" if table.caption else "Table"
    headers = table.columns

    for row in table.to_list():
        row_text = " | ".join(f"{headers[i]}: {row[i]}" for i in range(len(headers)))
        table_chunks.append(f"{table_caption} - {row_text}")
    
    return table_chunks

table_chunks = []
for table in tables:
    table_chunks.extend(convert_table_to_chunks(table))

# Combine all text and table chunks
all_chunks = text_chunks + table_chunks

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

# Step 5: Semantic Merging (Cross-Check Similarity between Text and Tables)
semantic_chunks = []
current_chunk = [tokenized_chunks[0]]
embeddings = model.encode(tokenized_chunks, convert_to_tensor=True)

for i in range(1, len(tokenized_chunks)):
    similarity = util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item()
    if similarity > 0.7:  # Adjust the threshold for semantic coherence
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
