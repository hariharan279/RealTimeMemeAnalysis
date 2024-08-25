import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import pytesseract

# Specify the full path to the Tesseract executable if needed
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path as necessary

# Define the label dictionary (same as during training)
label_dict = {
    'Appeal to (Strong) Emotions': 0,
    'Appeal to authority': 1,
    'Appeal to fear/prejudice': 2,
    'Bandwagon': 3,
    'Black-and-white Fallacy/Dictatorship': 4,
    'Causal Oversimplification': 5,
    'Doubt': 6,
    'Exaggeration/Minimisation': 7,
    'Flag-waving': 8,
    'Glittering generalities (Virtue)': 9,
    'Loaded Language': 10, 
    "Misrepresentation of Someone's Position (Straw Man)": 11,
    'Name calling/Labeling': 12,
    'Obfuscation, Intentional vagueness, Confusion': 13,
    'Presenting Irrelevant Data (Red Herring)': 14,
    'Reductio ad hitlerum': 15,
    'Repetition': 16,
    'Slogans': 17,
    'Smears': 18,
    'Thought-terminating cliche': 19,
    'Transfer': 20,
    'Whataboutism': 21
}

persuasion_techniques = [key for key in label_dict]

# Define the CLIPClassifier model class
class CLIPClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(CLIPClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        embedding_dim = self.clip.config.projection_dim
        self.classifier = torch.nn.Linear(embedding_dim * 2, num_labels)  # Assuming concatenation of embeddings

    def forward(self, input_ids, pixel_values, attention_mask):
        outputs = self.clip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        combined_features = torch.cat((outputs.text_embeds, outputs.image_embeds), dim=-1)
        logits = self.classifier(combined_features)
        return logits

# Load the trained model
model = CLIPClassifier(num_labels=22)
model.load_state_dict(torch.load('CLIP100_subtask2a.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the processor
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# Function to preprocess the image (grayscale and thresholding)
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        binary_image = image.point(lambda x: 0 if x < 140 else 255, '1')  # Apply thresholding
        return binary_image.convert("RGB")  # Convert back to RGB for CLIP model
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to perform OCR on the image and classify the meme
def classify_single_meme(image_path, model, processor, label_dict, threshold=0.4):
    # Preprocess the image
    image = preprocess_image(image_path)
    if image is None:
        print("Image preprocessing failed.")
        return []

    # Perform OCR on the image to extract text
    ocr_text = pytesseract.image_to_string(image).strip()
    print(f"OCR Extracted Text: {ocr_text}")

    # If OCR returns empty, handle it
    if not ocr_text:
        print("No text detected in the image.")
        return []

    # Preprocess inputs using the CLIP processor
    inputs = processor(text=[ocr_text], images=[image], return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move to GPU if available

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()

    # Determine which labels are predicted
    predicted_labels_indices = (probs > threshold).nonzero()[0]
    predicted_labels = [persuasion_techniques[idx] for idx in predicted_labels_indices]

    return predicted_labels

# Example usage
#image_path = "prop_meme_987.png"
#predicted_labels = classify_single_meme(image_path, model, processor, label_dict)
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})



@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    # Check if the request contains both image file and text
    if 'image' not in request.files:
        return jsonify({'error': 'Please provide image and text'}), 400

    image_file = request.files['image']
    print("Image file", image_file)
    print("Image file name", image_file.filename)
    #text = request.form['text']

    # Check if image file and text are not empty
    if image_file.filename == '':
        return jsonify({'error': 'Image file or text is empty'}), 400

    # Save the image file temporarily
    image_path = image_file.filename
    image_file.save(image_path)

    print("Image saved")

    try:
        # Detect emotions in the meme
        #print("I got the reuqest the text is ",text)
        predicted_labels = classify_single_meme(image_path, model, processor, label_dict)
        print("Ikkadiki vacha", predicted_labels)
        #predicted_labels = detect_meme_emotion(image_path, text)
        return jsonify({'predicted_labels': predicted_labels}), 200
    except Exception as e:
        print("Error entnatne",e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)

print(f'Predicted labels: {predicted_labels}')
