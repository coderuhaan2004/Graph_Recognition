import cv2
import numpy as np

# Load image
image = cv2.imread('graph.jpeg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize lists to store nodes and edges
nodes = []
edges = []

# Define a function to determine if a contour represents a node
def is_node(cnt):
    # Add conditions to determine if a contour represents a node
    # For example, based on contour area, aspect ratio, etc.
    return cv2.contourArea(cnt) > 100

# Iterate through contours
for cnt in contours:
    if is_node(cnt):
        # If contour represents a node, store its centroid coordinates
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            nodes.append((cX, cY))
    else:
        # If contour represents an edge, approximate its shape to a line
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        edges.append(approx)

# Initialize adjacency matrix
num_nodes = len(nodes)
adj_matrix = np.zeros((num_nodes, num_nodes))

# Define a function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Iterate through detected nodes and edges to populate adjacency matrix
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            node_i = nodes[i]
            node_j = nodes[j]
            min_distance = min(distance(node_i, edge[0][0]) for edge in edges)
            # If the minimum distance between nodes is below a threshold, consider them connected
            if min_distance < 20:  # Adjust threshold as needed
                adj_matrix[i][j] = 1

print("Adjacency Matrix:")
print(adj_matrix)
