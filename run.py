import warnings
import os
import torch
import clip
from scripts.clip_wrapper import ClipWrapper
from scripts.plot import visualisation_heatmap
from scripts.methods import vision_heatmap_ca_iba, text_heatmap_ca_iba
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from PIL import Image, ImageOps
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Script to run CA-M2IB on example paired image-text   
"""

def visualise(model, image, text, vision_beta=0.1, vision_var=1, vision_layer=9, text_beta=0.1, text_var=1, text_layer=9):
    # Preprocess image (3*224*224)
    image = Image.open(image).convert('RGB')
    image_features = processor(images=image, return_tensors="pt")['pixel_values'].to(device, dtype=torch.float) 
    # Tokenize text 
    ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device, dtype=torch.float)
    words = tokenizer.convert_ids_to_tokens(ids[0].tolist())
    # Train cross-attention information bottleneck on image
    print("CA-M2IB on image")
    vmap = vision_heatmap_ca_iba(ids, image_features, model, vision_layer, vision_beta, vision_var)
    # Train cross-attention information bottleneck on text
    print("CA-M2IB on text")
    tmap = text_heatmap_ca_iba(ids, image_features, model, text_layer, text_beta, text_var)
    image_processed = processor(images=image, return_tensors="pt", do_normalize=False)['pixel_values'][0].permute(1,2,0) # no normalization
    visualisation_heatmap(tmap, vmap, words, image_processed)
    
#Load pre-trained models    
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

# Example on Conceptual Dataset 
# Image-Paired Caption
img = "./images/cardinal.png" 
text_caption = "northern cardinal perched on a branch during a heavy winter snow storm."
# Run model and visualise heatmaps 
visualise(model, img, text_caption)