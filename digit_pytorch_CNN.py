from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tmp_dataset = datasets.MNIST(root = "./data", train = True, download = True,
                             transform = transforms.ToTensor())
tmp_loader = DataLoader(tmp_dataset, batch_size=len(tmp_dataset), shuffle = False)

images, _ = next(iter(tmp_loader)) # all 60,000 images
mean = images.mean()
std = images.std()

#preprocessing pipeline
train_tf = transforms.Compose([ 
    # small random rotations and translations
    transforms.RandomAffine(degrees = 5,
                            translate = (0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize((mean.item(),),(std.item(),)) # feature scaling where data points are adjusted to a common scale 
])

eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean.item(),),(std.item(),))
])


train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = train_tf,
    download = True
)
test_set = datasets.MNIST(
    root = 'data',
    train = False,
    transform = eval_tf,
    download = True
)

# 80/10/10 split for train, val, test data
g= torch.Generator().manual_seed(42)
train_set, val_set, _ = random_split(train_data, [int(0.80*len(train_data)), int(0.10*len(train_data)), int(0.10*len(train_data))], generator = g)
val_set.dataset.transform = eval_tf

BATCH = 256
loaders = {
    'train': DataLoader(train_set,
                        batch_size = BATCH,
                        shuffle = True,
                        num_workers = 2),
    'validation': DataLoader(val_set, 
                             batch_size = BATCH,
                             shuffle = False,
                             num_workers = 2),
    'test': DataLoader(test_set,
                       batch_size = BATCH, 
                       shuffle = False,
                       num_workers = 2),
}

class MNISTCNN(nn.Module): #subclass nn.Module to define a model
    def __init__(self):
        super().__init__()
        # in channels: 1 (since grayscale---if RGB this would be 3) 
        # in channels = number of channels in input image
        # out channels = 32 (number of filters)
        # kernel size = 3 (filter looks at 3x3 pixels at a time)
        self.conv1 = nn.Conv2d(1, 32, 3)
        # second layer takes in the 32 feature maps from first layer and puts through 64 filters
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        # drop 25% during training for regularization 
        self.drop1 = nn.Dropout(0.25)
        # dense: flattens features
        self.fc1 = nn.Linear(64*12*12, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x
    
def main():

    model = MNISTCNN().to(device) # creates model and moves to GPU/CPU
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    @torch.no_grad() # disables autograd to speed up

    def evaluate(loader):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader: # to iterate over all batches in val/test set
            x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True) # moves data to GPU/CPU where the model lives
            logits = model(x) # forward pass -> shape (N, 10)
            loss = criterion(logits, y) 

            total_loss += loss.item() * x.size(0) # sum loss weighted by batch size
            preds = logits.argmax(dim = 1) # predicted class index per example
            correct += (preds == y).sum().item()
            total += x.size(0)
        return total_loss /total, correct / total

    EPOCHS = 6
    for epoch in range(1, EPOCHS +1):
        model.train() 
        running_loss = 0.0

        for x, y in loaders['train']:
            x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True) # moves data to GPU/CPU where the model lives
            logits = model(x) # forward pass -> shape (N, 10)
            loss = criterion(logits, y) # compute batch loss

            optimizer.zero_grad() # clear old gradients
            loss.backward() # backprop, compute gradients via autograd
            optimizer.step() # update parameters

            running_loss += loss.item() * x.size(0) 

        train_loss = running_loss / len(loaders['train'].dataset)
        val_loss, val_acc = evaluate(loaders['validation'])
        train_eval_loss, train_acc = evaluate(loaders['train'])
        print("Epoch:", epoch, "| Train Loss:", train_loss, "Train Acc:", train_acc, "| Val Loss:", val_loss, "| Val Acc:", val_acc)

    test_loss, test_acc = evaluate(loaders['test'])
    print("Test Loss:", test_loss, "| Test Acc:", test_acc)
        
if __name__ == "__main__":
    main()