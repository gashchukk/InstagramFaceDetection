import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import networkx as nx
from matplotlib import pyplot as plt
from torchvision.models import squeezenet1_1

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            images.append(os.path.join(folder, filename))
    return images


class SimpleEmbeddingModel:
    def __init__(self):
        self.model = squeezenet1_1(pretrained=True)  # Load SqueezeNet1.1 pre-trained on ImageNet
        self.model.classifier = nn.Identity()  # Remove the classification layer
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def compute(self, face_image):
        face_tensor = self.transform(face_image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = self.model(face_tensor)  # Get the embedding
        return embedding.squeeze().numpy()  # Return as NumPy array


# Step 3: Detect faces and extract embeddings using OpenCV
def get_face_embeddings(image_path, face_cascade, model):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_data = []  # To store embeddings with bounding boxes and image paths

    for (x, y, w, h) in faces:
        # Extract face region
        face = image[y:y+h, x:x+w]
        face_embedding = model.compute(face)  # Get face embedding
        face_data.append({"embedding": face_embedding, "bbox": (x, y, w, h), "image_path": image_path})

    return face_data

# Step 4: Build the graph
def build_graph(face_data):
    graph = nx.Graph()
    for i, image_faces in enumerate(face_data):
        for face in image_faces:
            graph.add_node(face["embedding"].tobytes(), data=face)  # Add node with face data
        for j, other_image_faces in enumerate(face_data[i + 1:], start=i + 1):
            for face1 in image_faces:
                for face2 in other_image_faces:
                    # Use cosine similarity to compare embeddings
                    similarity = np.dot(face1["embedding"], face2["embedding"]) / (
                        np.linalg.norm(face1["embedding"]) * np.linalg.norm(face2["embedding"])
                    )
                    if similarity > 0.8:  # Threshold for similarity
                        graph.add_edge(face1["embedding"].tobytes(), face2["embedding"].tobytes())
    return graph

# Step 5: Calculate centrality
def calculate_centrality(graph):
    centrality = nx.degree_centrality(graph)
    most_influential = max(centrality, key=centrality.get)
    return graph.nodes[most_influential]["data"], centrality[most_influential]
# Step 8: Plot and save the graph
def plot_and_save_graph(graph, output_path="graph_visualization.png"):
    plt.figure(figsize=(10, 10))
    
    # Create a layout for the nodes
    pos = nx.spring_layout(graph, seed=42)  # Spring layout for better visualization
    
    # Draw the graph
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=50,
        node_color="blue",
        edge_color="gray",
        alpha=0.7
    )
    
    # Save the plot as an image
    plt.title("Face Similarity Graph", fontsize=16)
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()
    print(f"Graph saved at: {output_path}")

# Step 6: Main pipeline (Updated)
def main(images_folder):
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model = SimpleEmbeddingModel()

    images = load_images_from_folder(images_folder)
    face_data = []

    print("Processing images...")
    for image_path in images:
        data = get_face_embeddings(image_path, face_cascade, model)
        face_data.append(data)

    print("Building graph...")
    graph = build_graph(face_data)

    print("Calculating centrality...")
    most_influential, influence_score = calculate_centrality(graph)

    print("\nMost Influential Person:")
    print(f"Face ID: {most_influential}")
    print(f"Influence Score: {influence_score}")

    # Step 7: Display the image with the bounding box
    image_path = most_influential["image_path"]
    bbox = most_influential["bbox"]

    image = cv2.imread(image_path)
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, "Most Influential", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Most Influential Person", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result image
    output_path = "most_influential_person.jpg"
    cv2.imwrite(output_path, image)
    print(f"Image saved at: {output_path}")

    # Step 8: Plot and save the graph
    plot_and_save_graph(graph, output_path="face_similarity_graph.png")

if __name__ == "__main__":
    images_folder = "../inst/ucu_apps" 
    main(images_folder)
