from docling_core.types.doc import DoclingDocument, TextItem, TableItem, SectionHeaderItem
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import nltk
nltk.download('punkt')

# Initialize components
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Step 1: Load the document (replace with your document loading logic)
document = DoclingDocument.load("your_document.json")

# Step 2: Initialize lists for different item types
text_chunks = []
table_chunks = []

# Step 3: Process each item in the document
for item in document.items:
    if isinstance(item, TextItem):
        # Handle TextItem (paragraphs, list items)
        text_chunks.append(item.text)
    elif isinstance(item, SectionHeaderItem):
        # Handle SectionHeaderItem (add context to related text)
        text_chunks.append(f"Section: {item.text}")
    elif isinstance(item, TableItem):
        # Handle TableItem (convert tables into meaningful text chunks)
        table_caption = item.caption if item.caption else "Table"
        rows = item.to_list()

        # Detect table structure and convert rows
        if item.columns:
            # Standard table with headers
            headers = item.columns
            for row in rows:
                row_text = ", ".join(f"{headers[i]}: {row[i]}" for i in range(len(headers)))
                table_chunks.append(f"{table_caption} - {row_text}")
        elif len(rows[0]) == 2:
            # Two-column key-value table
            for row in rows:
                table_chunks.append(f"{table_caption} - {row[0]}: {row[1]}")
        else:
            # No headers, treat each row as a separate sentence
            for i, row in enumerate(rows):
                row_text = " | ".join(row)
                table_chunks.append(f"{table_caption} - Row {i + 1}: {row_text}")

# Step 4: Combine text and table chunks
all_chunks = text_chunks + table_chunks

# Step 5: Token-based chunking with overlap
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

# Step 6: Semantic Merging
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

# Step 7: Output final semantic chunks
print("Final Semantic Chunks:")
for idx, chunk in enumerate(semantic_chunks):
    print(f"Chunk {idx + 1}:\n{chunk}\n")
