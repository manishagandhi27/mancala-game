# %% [markdown]
# # Flowchart Parser & Relation Extractor (No OpenCV Version)
#
# This notebook demonstrates an alternative approach to parsing a flowchart image without using OpenCV.
# It leverages:
# - **scikit-image** for image processing (reading, thresholding, edge detection, Hough transform)
# - **Pillow (PIL)** for handling image regions for OCR
# - **pytesseract** for OCR text extraction
# - **matplotlib** for visualization
#
# Make sure Tesseract is installed on your system.
#
# ## Installation of Dependencies
# Run the cell below if you have not installed the required packages.

# %% [code]
# Install required pip packages (uncomment and run if needed)
# %pip install scikit-image pillow numpy matplotlib pytesseract

# %% [markdown]
# ## Imports

# %% [code]
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology, feature, transform
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from PIL import Image
import pytesseract

# If Tesseract is not in your PATH, uncomment and adjust the line below:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # for Linux
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # for Windows

# %% [markdown]
# ## Helper Function: Display an Annotated Image
#
# This function will be used later to show the annotated flowchart.

# %% [code]
def display_annotated_image(image, ax=None, title='Annotated Flowchart'):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
    plt.show()

# %% [markdown]
# ## Step 1: Detect Flowchart Nodes (Boxes)
#
# We convert the image to grayscale, threshold it, remove small objects, and then label connected regions.
# For each region (candidate node), if it meets size and shape criteria, we extract its bounding box and perform OCR.

# %% [code]
def detect_nodes(image):
    """
    Detects candidate flowchart nodes from the image.
    Uses scikit-image for thresholding and region labeling.
    
    Returns:
        A list of dictionaries:
          - 'bbox': (x, y, width, height)
          - 'text': OCR-extracted text from the region.
    """
    # Ensure the image is in float [0,1]
    if image.dtype != np.float32 and image.dtype != np.float64:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.copy()
    
    gray = color.rgb2gray(image_float)
    
    # Use Otsu's method to determine threshold
    thresh_val = filters.threshold_otsu(gray)
    binary = gray < thresh_val  # assuming dark text/lines on a lighter background
    
    # Remove small objects (noise)
    binary_clean = morphology.remove_small_objects(binary, min_size=150)
    
    # Label connected components
    labeled = measure.label(binary_clean)
    regions = measure.regionprops(labeled)
    
    nodes = []
    for region in regions:
        # Filter regions based on area and rectangular-like shape
        if region.area > 500:
            min_row, min_col, max_row, max_col = region.bbox
            width = max_col - min_col
            height = max_row - min_row
            if width > 30 and height > 30:
                # Extract the region of interest (ROI) and convert to a PIL image for OCR.
                roi = image[min_row:max_row, min_col:max_col]
                # If image is float, scale it to 0-255
                if roi.dtype != np.uint8:
                    roi_uint8 = (roi * 255).astype(np.uint8)
                else:
                    roi_uint8 = roi
                pil_roi = Image.fromarray(roi_uint8)
                # OCR extraction with Tesseract; psm 6 assumes a uniform block of text.
                config = '--psm 6'
                text = pytesseract.image_to_string(pil_roi, config=config).strip()
                nodes.append({'bbox': (min_col, min_row, width, height), 'text': text})
    return nodes

# %% [markdown]
# ## Step 2: Detect Arrowheads (Candidate Regions)
#
# We use Canny edge detection and then scikit-image's `measure.find_contours` to look for small contour regions
# that may correspond to arrowheads. (This is a heuristic approach; adjustments may be needed.)

# %% [code]
def detect_arrow_heads(image):
    """
    Detects candidate arrowhead locations using edge detection and contour analysis.
    
    Returns:
        A list of (x, y) tuples (pixel coordinates) for arrowhead candidates.
    """
    # Convert image to grayscale (using scikit-image, image assumed in [0,255] uint8)
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()
    gray = color.rgb2gray(image_uint8)
    
    # Use Canny edge detection
    edges = canny(gray, sigma=1.0)
    
    # Find contours at a constant value of 0.5
    contours = measure.find_contours(edges, level=0.5)
    arrow_heads = []
    for contour in contours:
        if len(contour) < 5:
            continue
        # Approximate bounding box for the contour
        min_row, min_col = np.min(contour, axis=0)
        max_row, max_col = np.max(contour, axis=0)
        area = (max_row - min_row) * (max_col - min_col)
        # Heuristically filter for small areas (likely arrowheads)
        if 10 < area < 500:
            centroid = np.mean(contour, axis=0)
            # Note: measure.find_contours returns (row, col), so swap to (x, y)
            arrow_heads.append((int(centroid[1]), int(centroid[0])))
    return arrow_heads

# %% [markdown]
# ## Step 3: Detect Arrows Using the Probabilistic Hough Transform
#
# We detect line segments using scikit-image's probabilistic Hough transform.
# Then, for each line, we check whether one of its endpoints is near an arrowhead candidate.
# If so, we designate that endpoint as the tip and the other as the tail.

# %% [code]
def detect_arrows(image, arrow_heads):
    """
    Detects arrow line segments and associates one endpoint with a nearby arrowhead.
    
    Returns:
        A list of tuples: (tail_point, tip_point) where each point is (x, y).
    """
    # Ensure image is in grayscale (0-1 float)
    if image.dtype != np.float32 and image.dtype != np.float64:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.copy()
    gray = color.rgb2gray(image_float)
    
    # Edge detection using Canny
    edges = canny(gray, sigma=1.0)
    
    # Use probabilistic Hough transform to detect line segments.
    # Note: The returned line endpoints are in (col, row) order.
    lines = probabilistic_hough_line(edges, threshold=10, line_length=30, line_gap=3)
    
    arrows = []
    for line in lines:
        (x1, y1), (x2, y2) = line  # (x, y) order
        tip = None
        tail = None
        # Check if either endpoint is near an arrowhead candidate (within 10 pixels)
        for (ax, ay) in arrow_heads:
            if np.hypot(x1 - ax, y1 - ay) < 10:
                tip = (x1, y1)
                tail = (x2, y2)
                break
            elif np.hypot(x2 - ax, y2 - ay) < 10:
                tip = (x2, y2)
                tail = (x1, y1)
                break
        if tip is not None and tail is not None:
            arrows.append((tail, tip))
    return arrows

# %% [markdown]
# ## Step 4: Assign Points to Nodes & Extract Relations
#
# For each detected arrow, we assign its tail and tip to the nearest node.
# If an endpoint lies inside a node's bounding box, it is associated with that node;
# otherwise, we choose the node whose center is closest.
#
# Relations are output as: "From [source text] to [destination text]".

# %% [code]
def assign_node(point, nodes):
    """
    Given a point (x, y) and a list of node dictionaries, returns the node that contains the point,
    or if not inside any node, the nearest node (by center distance).
    """
    x_pt, y_pt = point
    for node in nodes:
        x, y, w, h = node['bbox']
        if x <= x_pt <= x+w and y <= y_pt <= y+h:
            return node
    # Otherwise, find the closest node center
    min_dist = float('inf')
    assigned_node = None
    for node in nodes:
        x, y, w, h = node['bbox']
        cx = x + w / 2
        cy = y + h / 2
        dist = np.hypot(x_pt - cx, y_pt - cy)
        if dist < min_dist:
            min_dist = dist
            assigned_node = node
    return assigned_node

def extract_relations(arrows, nodes):
    """
    For each detected arrow, assigns its tail and tip to nodes and returns the extracted relations.
    
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
# This function loads the image, detects nodes, arrowheads, and arrows,
# extracts relations, and then visualizes the annotated image with detected elements.

# %% [code]
def parse_flowchart(image_path):
    # Load the image using scikit-image
    image = io.imread(image_path)
    if image is None:
        print("Error: Unable to load image. Check the path:", image_path)
        return None
    
    # For visualization, work on a copy
    annotated = image.copy()
    
    # Step 1: Detect nodes
    nodes = detect_nodes(image)
    print("Detected {} node(s).".format(len(nodes)))
    
    # Create a figure for annotation
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    for i, node in enumerate(nodes):
        x, y, w, h = node['bbox']
        # Draw a rectangle (using matplotlib.patches)
        rect = plt.Rectangle((x, y), w, h, edgecolor='green', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y-5, str(i+1), color='green', fontsize=12)
        print("Node {}: '{}' at {}".format(i+1, node['text'], node['bbox']))
    
    # Step 2: Detect arrowheads
    arrow_heads = detect_arrow_heads(image)
    print("Detected {} arrowhead candidate(s).".format(len(arrow_heads)))
    for (ax_val, ay_val) in arrow_heads:
        ax.plot(ax_val, ay_val, 'bo')  # blue dot
    
    # Step 3: Detect arrows using Hough transform
    arrows = detect_arrows(image, arrow_heads)
    print("Detected {} arrow(s).".format(len(arrows)))
    for arrow in arrows:
        tail, tip = arrow
        ax.annotate("", xy=tip, xytext=tail,
                    arrowprops=dict(arrowstyle="->", color='red', lw=2))
    
    # Step 4: Extract relations from arrows and nodes
    relations = extract_relations(arrows, nodes)
    print("\nExtracted Relations:")
    for rel in relations:
        print("From '{}' to '{}'".format(rel['source'], rel['destination']))
    
    ax.set_title("Annotated Flowchart")
    ax.axis('off')
    plt.show()
    
    return relations

# %% [markdown]
# ## Step 6: Run the Parser on a Flowchart Image
#
# Replace `'flowchart.png'` with the path to your flowchart image.
# The parser will print detected nodes, arrow relations, and display an annotated image.

# %% [code]
# Example usage:
relations = parse_flowchart('flowchart.png')
