### CLIP 


![CLIP img](https://production-media.paperswithcode.com/methods/3d5d1009-6e3d-4570-8fd9-ee8f588003e7.png)

[CLIP](https://openai.com/index/clip/) (Contrastive Language-Image Pretraining) is a model developed by OpenAI that learns to align images and textual descriptions in a shared embedding space using contrastive learning. It trains by maximizing the similarity between correct image-text pairs and minimizing it for incorrect pairs, effectively learning rich, transferable representations of both modalities. Unlike traditional supervised learning, CLIP does not require labeled datasets but rather leverages large-scale image-text pairs to perform well on various vision-and-language tasks.

### Important Parts:
- Image Encoder (can be [MAE](https://arxiv.org/pdf/2111.06377))
- Text Encoder ( can be BERT or CLIP's own text Encoder)
- Loss Function (can be NTXentLoss)

### Code template

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


### Functions to implement
Even with current Lightly settings it is already possible to intergrate and train CLIP. To improve flexibility we can implement Text Encoders in the library and more Contrastive Losses. Also, we can modify Dataset creation process so that we will be able to transform (tokenize) target.
## Text Encoder
Text Encoder can be implemented in the same way as [MAE](https://github.com/Katerina5649/lightly/blob/master/lightly/models/modules/masked_autoencoder.py).
## Loss Functions
#### InfoNCE : 

$$\text{Loss}\_{\text{contrastive}} = - \frac{1}{N} \sum_{i=1}^N \log \frac{\exp\(\text{sim}\(x_i, y_i\) / \tau\)}{\sum_{j=1}^N \exp\(\text{sim}\(x_i, y_j\) / \tau\)}$$

where:
- $\text{sim}(x_i, y_j)$ denotes the cosine similarity between the image embedding $x_i$ and text embedding $y_j$.
- $\tau$ is the temperature parameter that scales the similarity values.
- $N$ is the number of image-text pairs in a batch.

#### Triplet Loss:
The triplet loss is a widely used contrastive loss function that operates on triplets of anchor, positive, and negative examples. The goal is to maximize the distance between the anchor and negative examples while minimizing the distance between the anchor and positive examples. The triplet loss is defined as:

$$L = \max(d(a, p) - d(a, n) + \alpha, 0)$$

where $d(a, p)$ is the distance between the anchor and positive example, $d(a, n)$ is the distance between the anchor and negative example, and $\alpha$ is a margin parameter.

#### Multi-Class N-Pair Loss:
This loss function is an extension of the triplet loss to multiple positive and negative examples. It is defined as:

$$L = \log\left(1 + \sum_{p=1}^{P}\sum_{n=1}^{N}\exp(s(x, x_n^-) - s(x, x_p^+))\right)$$

where $s(x, x_p^+)$ is the similarity between the anchor and the $p$-th positive example, $s(x, x_n^-)$ is the similarity between the anchor and the $n$-th negative example, and $P$ and $N$ are the number of positive and negative examples, respectively.
### Datasets
Right now my CLIP's code calls ```text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")``` every epoch which is not suffisient. We can add one more argument to LightlyDataset ```target_trasform``` to transorm target before training.
