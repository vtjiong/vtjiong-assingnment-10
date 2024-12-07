from flask import Flask, render_template,jsonify,request
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import open_clip
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import os

app = Flask(__name__)


def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names


def nearest_neighbors(query_embedding, embeddings, top_k=5):
    distances = euclidean_distances(query_embedding.reshape(1, -1), embeddings).flatten()
    nearest_indices = np.argsort(distances)[:top_k]
    return nearest_indices, distances[nearest_indices]


train_images, train_image_names = load_images("./static/coco_images_resized", max_images=2000,
                                              target_size=(224, 224))
transform_images, transform_image_names = load_images("./static/coco_images_resized", max_images=10000,
                                                      target_size=(224, 224))


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    request_type = request.form.get('query-type')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
    df = pd.read_pickle('image_embeddings.pickle')
    if request_type == "image":
        image = request.files.get("image-query")
        if request.form.get('numComponents'):
            global train_images
            global train_image_names
            global transform_images
            global transform_image_names
            num_components = request.form.get('numComponents')
            k = int(num_components)
            pca = PCA(n_components=k)
            pca.fit(train_images)
            reduced_embeddings = pca.transform(transform_images)
            # Preprocess uploaded image
            query_image = Image.open(image)
            query_image = query_image.convert('L')  # Convert to grayscale
            query_image = query_image.resize((224, 224))  # Resize to target size
            query_image_array = np.asarray(query_image, dtype=np.float32).flatten() / 255.0  # Normalize and flatten
            # Transform query image using PCA
            query_embedding = pca.transform([query_image_array])[0]
            # Use nearest_neighbors function
            top_indices, top_distances = nearest_neighbors(query_embedding, reduced_embeddings, top_k=5)
            top_results = [
                {"file_name": os.path.join("/static/coco_images_resized", transform_image_names[idx]), "distances": top_distances[i]}
                for i, idx in enumerate(top_indices)
            ]
            return jsonify({"status": "success", "results": top_results})

        image_processed = preprocess(Image.open(image)).unsqueeze(0)
        query_embedding = F.normalize(model.encode_image(image_processed))
        # Convert the query embedding to a numpy array
        query_embedding_np = query_embedding.detach().cpu().numpy()
        # Extract embeddings from the DataFrame and stack them as a numpy array
        embeddings_np = np.vstack(df['embedding'].to_numpy())
        # Compute cosine similarities between the query embedding and the dataset embeddings
        similarities = cosine_similarity(query_embedding_np, embeddings_np)
        top_n = 5
        top_indices = np.argsort(similarities[0])[::-1][:top_n]
        top_results = [
            {"file_name": f"/static/coco_images_resized/{df.iloc[i]['file_name']}", "similarity": float(similarities[0][i])}
            for i in top_indices
        ]
        return jsonify({"status": "success", "message": "Image query processed successfully.", "results": top_results})
    elif request_type == "text":
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text = request.form.get("text-query")
        model.eval()
        text = tokenizer([text])
        query_embedding = F.normalize(model.encode_text(text))
        # Retrieve the image path that corresponds to the embedding in `df`
        # with the highest cosine similarity to query_embedding
        # Convert the query embedding to a numpy array
        query_embedding_np = query_embedding.detach().cpu().numpy()
        # Extract embeddings from the DataFrame and stack them as a numpy array
        embeddings_np = np.vstack(df['embedding'].to_numpy())
        # Compute cosine similarities between the query embedding and the dataset embeddings
        similarities = cosine_similarity(query_embedding_np, embeddings_np)
        top_n = 5
        top_indices = np.argsort(similarities[0])[::-1][:top_n]
        top_results = [
            {"file_name": f"/static/coco_images_resized/{df.iloc[i]['file_name']}", "similarity": float(similarities[0][i])}
            for i in top_indices
        ]
        return jsonify({"status": "success", "message": "Image query processed successfully.", "results": top_results})
    else:
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        image_data = request.files.get('image-query')
        text_data = request.form.get('text-query')
        weight = request.form.get('hybrid-query')
        weight = float(weight)
        image = preprocess(Image.open(image_data)).unsqueeze(0)
        image_query = F.normalize(model.encode_image(image))
        model.eval()
        text = tokenizer([text_data])
        text_query = F.normalize(model.encode_text(text))
        lam = weight  # tune this
        query = F.normalize(lam * text_query + (1.0 - lam) * image_query)
        # Convert the hybrid query embedding to a numpy array
        query_embedding_np = query.detach().cpu().numpy()
        # Extract embeddings from the DataFrame and stack them as a numpy array
        embeddings_np = np.vstack(df['embedding'].to_numpy())
        similarities = cosine_similarity(query_embedding_np, embeddings_np)
        top_n = 5
        top_indices = np.argsort(similarities[0])[::-1][:top_n]
        top_results = [
            {"file_name": f"/static/coco_images_resized/{df.iloc[i]['file_name']}", "similarity": float(similarities[0][i])}
            for i in top_indices
        ]
        return jsonify({"status": "success", "message": "Image query processed successfully.", "results": top_results})


if __name__ == '__main__':
    app.run(debug=True, port=3000)
