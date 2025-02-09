# %% [markdown]
# # Flowchart Parser & Relation Extractor Notebook
#
# This notebook demonstrates a heuristic approach to:
#
# 1. **Extracting Flowchart Nodes:**  
#    Detect rectangular regions (nodes) in a flowchart and extract their text using Tesseract OCR.
#
# 2. **Detecting Arrows:**  
#    Detect arrowheads via contour analysis and use a Hough Line Transform to find arrow lines.
#
# 3. **Extracting Relations:**  
#    Associate arrow endpoints with the closest nodes to build “source → destination” relations.
#
# **Note:**
# - Ensure you have Tesseract installed (e.g., on Ubuntu: `sudo apt-get install tesseract-ocr`).  
# - If Tesseract is not in your PATH, set its location by modifying `pytesseract.pytesseract.tesseract_cmd`.
#
# ## Installation of Dependencies
# Run the following cell to install required Python packages.

# %% [code]
# Install required pip packages (uncomment and run if needed)
# !pip install opencv-python-headless numpy pytesseract matplotlib

# %% [markdown]
# ## Import Required Libraries

# %% [code]
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

# Uncomment and adjust the following line if Tesseract is not in your PATH.
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# %% [markdown]
# ## Helper Function: Display an Image
#
# This function displays an image using matplotlib.

# %% [code]
def display_image(img, title='Image', figsize=(10, 10)):
    """Display an image (converting BGR to RGB) using matplotlib."""
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# %% [markdown]
# ## Step 1: Detect Flowchart Nodes (Boxes)
#
# We convert the image to grayscale, apply adaptive thresholding and dilation, and then find contours.
# Rectangular contours (with four approximate vertices) and large enough dimensions are assumed to be nodes.
# Each node's ROI is processed with Tesseract to extract text.

# %% [code]
def detect_nodes(image):
    """
    Detects rectangular nodes in the flowchart image and extracts text from each.
    
    Returns:
        A list of dictionaries with keys:
         - 'bbox': (x, y, w, h)
         - 'text': OCR-extracted text from the ROI.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding to account for variable illumination
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # Dilate to merge nearby text/lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nodes = []
    for cnt in contours:
        # Approximate contour to a polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:  # likely a rectangle (node)
            x, y, w, h = cv2.boundingRect(approx)
            if w > 30 and h > 30:  # filter out very small regions
                roi = image[y:y+h, x:x+w]
                # Using Tesseract with psm 6 (assumes a uniform block of text)
                config = '--psm 6'
                text = pytesseract.image_to_string(roi, config=config)
                nodes.append({'bbox': (x, y, w, h), 'text': text.strip()})
    return nodes

# %% [markdown]
# ## Step 2: Detect Arrowheads
#
# We use Canny edge detection and contour analysis to find small triangular shapes
# that are likely arrowheads.

# %% [code]
def detect_arrow_heads(image):
    """
    Detect candidate arrowhead locations in the image.
    
    Returns:
        A list of (x, y) tuples representing the centroids of arrowhead candidates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    arrow_heads = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by area; tune these values based on your image characteristics
        if 10 < area < 500:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 3:  # triangle candidate
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    arrow_heads.append((cx, cy))
    return arrow_heads

# %% [markdown]
# ## Step 3: Detect Arrows Using Hough Transform and Associate with Arrowheads
#
# We detect line segments using the Hough Transform. If one endpoint of a line is near an arrowhead candidate,
# we treat that endpoint as the tip (arrowhead) and the other endpoint as the tail.

# %% [code]
def detect_arrows(image, arrow_heads):
    """
    Detects arrow line segments and associates one endpoint with a nearby arrowhead.
    
    Returns:
        A list of tuples: (tail_point, tip_point) where each point is (x, y).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Probabilistic Hough Transform to detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    arrows = []
    if lines is None:
        return arrows
    for line in lines:
        x1, y1, x2, y2 = line[0]
        tip = None
        tail = None
        # Check if either endpoint is near one of the arrowhead candidates (within 10 pixels)
        for (ax, ay) in arrow_heads:
            if np.sqrt((x1 - ax)**2 + (y1 - ay)**2) < 10:
                tip = (x1, y1)
                tail = (x2, y2)
                break
            elif np.sqrt((x2 - ax)**2 + (y2 - ay)**2) < 10:
                tip = (x2, y2)
                tail = (x1, y1)
                break
        if tip is not None and tail is not None:
            arrows.append((tail, tip))
    return arrows

# %% [markdown]
# ## Step 4: Assign Points to Nodes & Extract Relations
#
# For each arrow, we assign its tail and tip to the nearest nodes.
# If an endpoint lies inside a node’s bounding box, it is assigned to that node.
# Otherwise, the nearest node (by center distance) is chosen.
# Relations are then extracted in the form: "From [source text] to [destination text]".

# %% [code]
def assign_node(point, nodes):
    """
    Given a point (x, y) and a list of node dictionaries, return the node that contains the point,
    or if not inside any node, the nearest node by center distance.
    """
    x_pt, y_pt = point
    for node in nodes:
        x, y, w, h = node['bbox']
        if x <= x_pt <= x+w and y <= y_pt <= y+h:
            return node
    # If the point isn't inside any node, pick the nearest node center
    min_dist = float('inf')
    assigned_node = None
    for node in nodes:
        x, y, w, h = node['bbox']
        cx = x + w / 2
        cy = y + h / 2
        dist = np.sqrt((x_pt - cx)**2 + (y_pt - cy)**2)
        if dist < min_dist:
            min_dist = dist
            assigned_node = node
    return assigned_node

def extract_relations(arrows, nodes):
    """
    For each detected arrow, assign its tail and tip to nodes and return the extracted relations.
    
    Returns:
         A list of dictionaries with keys:
            - 'source': Text from the source node.
            - 'destination': Text from the destination node.
            - 'arrow': The (tail, tip) tuple.
    """
    relations = []
    for arrow in arrows:
        tail, tip = arrow
        src_node = assign_node(tail, nodes)
        dst_node = assign_node(tip, nodes)
        if src_node is not None and dst_node is not None and src_node != dst_node:
            relations.append({
                'source': src_node['text'] if src_node['text'] else "Unknown",
                'destination': dst_node['text'] if dst_node['text'] else "Unknown",
                'arrow': arrow
            })
    return relations

# %% [markdown]
# ## Step 5: Main Function to Process the Flowchart Image
#
# This function loads an image, processes it to detect nodes, arrowheads, arrows,
# extracts relations, and then displays an annotated image.
#
# You can also print out the extracted relations.

# %% [code]
def parse_flowchart(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image. Please check the path:", image_path)
        return None
    original = image.copy()
    
    # Step 1: Detect nodes (boxes)
    nodes = detect_nodes(image)
    print("Detected {} node(s).".format(len(nodes)))
    for i, node in enumerate(nodes):
        print("Node {}: '{}' at {}".format(i+1, node['text'], node['bbox']))
        x, y, w, h = node['bbox']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Step 2: Detect arrowheads
    arrow_heads = detect_arrow_heads(original)
    print("Detected {} arrowhead candidate(s).".format(len(arrow_heads)))
    for (ax, ay) in arrow_heads:
        cv2.circle(image, (ax, ay), 5, (255, 0, 0), -1)
    
    # Step 3: Detect arrows using Hough Transform
    arrows = detect_arrows(original, arrow_heads)
    print("Detected {} arrow(s).".format(len(arrows)))
    for arrow in arrows:
        tail, tip = arrow
        cv2.arrowedLine(image, tail, tip, (0, 0, 255), 2, tipLength=0.2)
    
    # Step 4: Extract relations from detected arrows and nodes
    relations = extract_relations(arrows, nodes)
    print("\nExtracted Relations:")
    for rel in relations:
        print("From '{}' to '{}'".format(rel['source'], rel['destination']))
    
    # Display the annotated image
    display_image(image, title="Annotated Flowchart")
    
    return relations

# %% [markdown]
# ## Step 6: Run the Parser on a Flowchart Image
#
# Replace `'flowchart.png'` with the path to your flowchart image.
# The parser will print detected nodes, arrow relations, and display an annotated image.

# %% [code]
# Example usage: (Ensure you have a suitable flowchart image in the working directory)
relations = parse_flowchart('flowchart.png')
