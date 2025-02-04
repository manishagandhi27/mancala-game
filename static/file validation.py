import hashlib
import re
import psycopg2

# Database Connection
conn = psycopg2.connect(
    dbname="your_db",
    user="your_user",
    password="your_password",
    host="your_host",
    port="your_port"
)
cursor = conn.cursor()

# File Name Validation Regex (Modify as Needed)
FILENAME_REGEX = r"^[a-zA-Z0-9_-]+\.(pdf|txt|docx)$"

def validate_filename(file_name):
    """Validates file name using regex."""
    return re.match(FILENAME_REGEX, file_name) is not None

def get_file_size_and_hash(file_obj, algo="sha256", chunk_size=8192):
    """Computes file hash and size in one pass."""
    hash_func = hashlib.new(algo)
    file_size = 0

    current_pos = file_obj.tell()  # Save current position
    file_obj.seek(0)  # Move to start of file

    while chunk := file_obj.read(chunk_size):
        hash_func.update(chunk)
        file_size += len(chunk)

    file_obj.seek(current_pos)  # Restore position
    return file_size, hash_func.hexdigest()

def check_file_existence(file_hash, file_name):
    """
    Checks if file hash or file name already exists in DB.
    Returns specific error messages based on the match.
    """
    query = """
        SELECT 
            CASE 
                WHEN file_hash = %s THEN 'hash_match'
                WHEN file_name = %s THEN 'name_match'
            END AS match_type
        FROM files
        WHERE file_hash = %s OR file_name = %s
        LIMIT 1;
    """
    cursor.execute(query, (file_hash, file_name, file_hash, file_name))
    result = cursor.fetchone()

    if result:
        return result[0]  # 'hash_match' or 'name_match'
    return None  # No match

def process_file(file_obj, file_name):
    """Validates file and prepares it for DB insertion."""
    
    # Step 1: Perform Regex Validation First
    if not validate_filename(file_name):
        return {"error": "Invalid file name format"}

    # Step 2: Compute File Size & Hash in One Pass
    file_size, file_hash = get_file_size_and_hash(file_obj)

    # Step 3: Single Query for Both File Hash & Name
    match_type = check_file_existence(file_hash, file_name)

    if match_type == "hash_match":
        return {"error": "Duplicate file detected (same content exists)"}
    elif match_type == "name_match":
        return {"error": "Duplicate file name detected (rename the file)"}

    # Step 4: Insert into DB Only If Valid
    insert_query = """
        INSERT INTO files (file_name, file_size, file_hash) VALUES (%s, %s, %s)
    """
    cursor.execute(insert_query, (file_name, file_size, file_hash
