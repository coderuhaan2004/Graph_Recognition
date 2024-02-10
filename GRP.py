import cv2
import numpy as np
image =  cv2.imread("graph.jpeg")
#Convert to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

#Apply thresholding
_, thresholded_image = cv2.threshold(grayscale_image,130,255,cv2.THRESH_BINARY)

#Find Contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

#Draw edge mask
edge_contours = []
for contour in contours:
    edge_contours.append(contour)
edge_mask = np.zeros_like(grayscale_image)
cv2.drawContours(edge_mask, edge_contours, -1, 255, thickness=1)
#negative_edge_mask = cv2.bitwise_not(edge_mask)

# Hough Circle Transform
circles = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30)

detected_nodes = []
node_mask = np.ones_like(grayscale_image)
centres = []

if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        # Draw detected circles
        center = (circle[0], circle[1])
        centres.append(center)
        radius = circle[2]
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.circle(node_mask,center,radius, (255, 0, 0), 5)
        detected_nodes.append(center)

def distance(p1,p2):
    dist = np.sqrt((p1[0]-p2[0])**2 + (p2[1]-p2[1])**2)
    return dist



num_nodes = len(centres)
adj_matrix = np.zeros((num_nodes,num_nodes))

for i in range(num_nodes):
    for j in range(num_nodes):
        if i!=j:
            node_i = centres[i]
            node_j = centres[j]
            min_distance = (distance(node_i,node_j))
            if min_distance < 20:
                adj_matrix[i][j] = 1

print("Adjacency Matrix: ")
print("x-coordinate of 4th node is: ", centres[3][0])
print(adj_matrix)
'''
417
'''

cv2.imshow("Thresh",node_mask)

print(centres)

'''
matrix = np.subtract(grayscale_image,negative_edge_mask)
cv2.imshow("graph",matrix)
'''

#node_mask = np.zeros_like(grayscale_image)
cv2.imwrite('edge_mask.png', edge_mask)
cv2.imwrite('node_mask.png', node_mask)
'''# Separate contours into edges and nodes
edge_contours = []
for contour in contours:
    # Calculate the area of each contour
    area = cv2.contourArea(contour)
    # If the area is below a certain threshold, consider it as a node contour
    if area > 500:
        edge_contours.append(contour)

# Create separate masks for edges and nodes
edge_mask = np.zeros_like(grayscale_image)
#node_mask = np.zeros_like(grayscale_image)
cv2.drawContours(edge_mask, edge_contours, -1, 255, thickness=cv2.FILLED)

# Invert the edge mask
#edge_mask = cv2.bitwise_not(edge_mask)

# Write edge and node masks to separate image files
cv2.imwrite('edge_mask.png', edge_mask)
'''

# Display the processed image
cv2.imshow('Processed Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('processed_image.png', image)
