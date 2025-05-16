import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel, DistilBertModel, XLNetModel
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer, XLNetTokenizer
from torchvision import models, transforms
from PIL import Image

# Define the Late Fusion Classifier
class LateFusionClassifier(nn.Module):
    def __init__(self, input_dim=7424):
        super(LateFusionClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load transformer models and move to device
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased').to(device)

# Load CNN models and modify for feature extraction
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()
resnet = resnet.to(device)

efficientnet = models.efficientnet_b0(pretrained=True)
efficientnet.classifier[1] = nn.Identity()
efficientnet = efficientnet.to(device)

densenet = models.densenet121(pretrained=True)
densenet.classifier = nn.Identity()
densenet = densenet.to(device)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# Extract image features
def extract_image_features(image_input):
    image = Image.open(image_input).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        resnet_features = resnet(image_tensor).squeeze(0).cpu()
        efficientnet_features = efficientnet(image_tensor).squeeze(0).cpu()
        densenet_features = densenet(image_tensor).squeeze(0).cpu()

    image_features = torch.cat([resnet_features, efficientnet_features, densenet_features], dim=0)
    return image_features

# Extract text features
def extract_text_features(text_input):
    inputs_bert = bert_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(device)
    inputs_roberta = roberta_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(device)
    inputs_distilbert = distilbert_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(device)
    inputs_xlnet = xlnet_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        bert_output = bert_model(**inputs_bert).last_hidden_state.mean(dim=1).squeeze(0).cpu()
        roberta_output = roberta_model(**inputs_roberta).last_hidden_state.mean(dim=1).squeeze(0).cpu()
        distilbert_output = distilbert_model(**inputs_distilbert).last_hidden_state.mean(dim=1).squeeze(0).cpu()
        xlnet_output = xlnet_model(**inputs_xlnet).last_hidden_state.mean(dim=1).squeeze(0).cpu()

    text_features = torch.cat([bert_output, roberta_output, distilbert_output, xlnet_output], dim=0)
    return text_features

# Prediction function
def late_fusion_predict(image_input=None, text_input=None):
    image_features = None
    text_features = None

    if image_input:
        image_features = extract_image_features(image_input)
        text_features = torch.zeros(3072)  # Dummy text features

    elif text_input:
        text_features = extract_text_features(text_input)
        image_features = torch.zeros(4352)  # Dummy image features

    if image_features is not None and text_features is not None:
        fused_features = torch.cat([text_features, image_features], dim=0).unsqueeze(0).to(device)
        model = LateFusionClassifier().to(device)
        model.load_state_dict(torch.load("late_fusion_model.pth", map_location=device))
        model.eval()

        with torch.no_grad():
            output = model(fused_features)
            probs = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            return predicted_class, probs

    return None, None  # If neither image nor text is provided
