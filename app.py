from flask import Flask, request, jsonify, render_template
from facenet_pytorch import InceptionResnetV1
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Simulated database (store one face per user)
DATABASE = "faces/"
os.makedirs(DATABASE, exist_ok=True)

# Function to get embeddings
def get_embedding(image_path):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    embedding = model(img_tensor)
    return embedding

# Save user face
@app.route('/register_face', methods=['POST'])
def register_face():
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID required"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join(DATABASE, f"{user_id}.jpg")
    file.save(file_path)
    return jsonify({"message": "Face registered successfully!"})

# Verify face
@app.route('/verify_face', methods=['POST'])
def verify_face():
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID required"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    file_path = "temp.jpg"
    file.save(file_path)

    # Check if user face exists
    user_face_path = os.path.join(DATABASE, f"{user_id}.jpg")
    if not os.path.exists(user_face_path):
        return jsonify({"error": "User face not found"}), 404

    # Compute embeddings
    stored_embedding = get_embedding(user_face_path)
    input_embedding = get_embedding(file_path)

    # Compute similarity
    similarity = torch.nn.functional.cosine_similarity(stored_embedding, input_embedding)
    THRESHOLD = 0.6
    result = "face_id_correct" if similarity > THRESHOLD else "face_id_incorrect"

    return jsonify({"user_id": user_id, "result": result, "similarity": similarity.item()})

@app.route('/')
def home():
    return render_template('test.html')



# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
