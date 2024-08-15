### CLIP 


![CLIP img](https://production-media.paperswithcode.com/methods/3d5d1009-6e3d-4570-8fd9-ee8f588003e7.png)

[CLIP](https://openai.com/index/clip/) (Contrastive Language-Image Pretraining) is a model developed by OpenAI that learns to align images and textual descriptions in a shared embedding space using contrastive learning. It trains by maximizing the similarity between correct image-text pairs and minimizing it for incorrect pairs, effectively learning rich, transferable representations of both modalities. Unlike traditional supervised learning, CLIP does not require labeled datasets but rather leverages large-scale image-text pairs to perform well on various vision-and-language tasks.

```python
import torch
import torchvision
from transformers import BertModel, BertTokenizer
from lightly import loss, transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads

# Define the CLIP Model
class CLIPModel(torch.nn.Module):
    def __init__(self, text_encoder, image_encoder, text_projection, image_projection):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.text_projection = text_projection
        self.image_projection = image_projection

    def forward(self, text_input, image_input):
        text_features = self.text_encoder(**text_input).pooler_output
        text_features = self.text_projection(text_features)
        image_features = self.image_projection(self.image_encoder(image_input))
        return text_features, image_features

# Load Pretrained BERT and MAE
text_encoder = ...# Example : BertModel.from_pretrained('bert-base-uncased')
tokenizer = ...# Example : BertTokenizer.from_pretrained('bert-base-uncased')

image_encoder = lightly.models.modules.MAEEncoder(...)

# Define projection heads
text_projection = torch.nn.Linear(text_encoder.config.hidden_size, 128)
image_projection = torch.nn.Linear(image_encoder.head.in_features, 128)

# Build the CLIP model
model = CLIPModel(text_encoder, image_encoder, text_projection, image_projection)

# Prepare transforms for images
transform = transforms.MAETransform()

# Create dataset (modify with your actual dataset path)
dataset = LightlyDataset(input_dir="./my/image/text/dataset/", transform=transform)

# Build a PyTorch dataloader
dataloader = torch.utils.data.DataLoader(dataset,...)

# Define a contrastive loss function
criterion = loss.NTXentLoss(temperature=0.5)

# Get a PyTorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        # Assume batch contains images and corresponding text strings
        images, texts = batch

        # Tokenize the texts
        text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Move tensors to GPU if available
        images = images.to('cuda') if torch.cuda.is_available() else images
        text_inputs = {key: val.to('cuda') if torch.cuda.is_available() else val for key, val in text_inputs.items()}
        model = model.to('cuda') if torch.cuda.is_available() else model

        # Forward pass
        text_features, image_features = model(text_inputs, images)

        # Compute loss
        loss = criterion(text_features, image_features)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

```
