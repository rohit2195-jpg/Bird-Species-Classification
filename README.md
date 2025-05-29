# Bird-Species-Classification

## Overview
Ths project uses a Convolutional Nueral Netowrk to identify bird species from a given image.
The model was trained on over 400 species of North American birds, with the NaBirds Dataset.
The model used a pretrained resnet50 architecture from pytorch library.

## Results
Data from the dataset was split into test, train, and validation sets.
The model currently has a 55% accuracy on the test set.

## Testing

1. Start the backend of the project
    ```bash
   cd backend
   python backend.py
2. Start the frontend by runnnig index.html
