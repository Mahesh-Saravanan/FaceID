import os
import torch

class Config:
    
    data_dir = "/Users/maheshsaravanan/Desktop/FaceID/DS"

    
    image_size = 512    #     
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-3
    latent_dim = 512
    num_workers = 4
    checkpoint_path ="checkpoints"
    SaveInterval = 2
    image_path = "/Users/maheshsaravanan/Desktop/FaceID/TestFolder/test1.jpg"

    Load_model = True
    Encoder_path = "./checkpoints/encoder_at_Epoch5.pth"
    Decoder_path = "./checkpoints/decoder_at_Epoch5.pth"
   
    split_ratios = (0.85, 0.10, 0.05)
    device ="mps" if torch.backends.mps.is_available() else "cpu"
