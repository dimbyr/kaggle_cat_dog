from dataset import cat_and_dog
from models import resnext101
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm
from train_test_functions import train, test

# =============================================================================
#  Load the data:   
# =============================================================================

train_data = cat_and_dog(
    path = '..' 
)
val_data = cat_and_dog(
    path = '..',
    phase= 'val'
)
test_data = cat_and_dog(path = '..',
                            phase= 'test')

print(f'{len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test data')
# =============================================================================
# Use torch.utils.data.DataLoader to match the requirement of the model
# =============================================================================

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data,
     batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data,
     batch_size=1000, shuffle=True)

# =============================================================================
#  Training settings 
# =============================================================================
model = resnext101(depth=8)
optimizer  = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
train(model, criterion, train_loader, optimizer, 5)
test(model, val_loader)