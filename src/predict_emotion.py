import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(__file__))
from train_cnn import Net

EMOTION_LABELS = {
    0: 'angry',
    1: 'disgust', 
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

def predict_emotion(features, model_path='model.pth'):
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location='cpu')) # loads pre-trained weights from model file
    model.eval() # set model to evaluation mode
    
    if len(features.shape) == 2:
        features = features.unsqueeze(0).unsqueeze(0)  # (1,1,48,48)
    elif len(features.shape) == 3:
        features = features.unsqueeze(0)
    
    with torch.no_grad(): # disable gradient comp
        outputs = model(features) # get raw model prediction
        probabilities = F.softmax(outputs, dim=1) # convert to probabilities
        confidence, predicted = torch.max(probabilities, 1) # gets highest probaibility
        
    emotion = EMOTION_LABELS[predicted.item()] # converts class to emotion name
    confidence_score = confidence.item()
    
    return {
        'emotion': emotion,
        'confidence': confidence_score
    }