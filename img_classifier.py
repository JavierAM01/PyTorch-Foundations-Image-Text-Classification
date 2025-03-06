import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse

import wandb
# wandb.login("API_KEY")

# CREATE A SMALL MAP FOR THE CAPTION OF THE IMAGES
mapping = {0: "parrot", 1: "narwhal", 2: "axolotl"}


img_size = (256,256)
num_labels = 3
grayscale = False

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)



class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        img_name = self.data_frame.loc[idx, "image"]
        image = Image.open(img_name).convert("RGB")  # Assuming RGB images
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data(batch_size):
    transform_img = T.Compose([
        T.ToTensor(), 
        T.Grayscale() if grayscale else T.Lambda(lambda x : x),   # change just if we need grayscale
        T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
        T.CenterCrop(img_size),  # Center crop to 256x256
        T.Normalize(mean=[0.5], std=[0.5]) if grayscale else 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize each color dimension
    ])
    train_data = CsvImageDataset(
        csv_file='./data/img_train.csv',
        transform=transform_img,
    )
    val_data = CsvImageDataset(
        csv_file='./data/img_val.csv',
        transform=transform_img,
    )
    test_data = CsvImageDataset(
        csv_file='./data/img_test.csv',
        transform=transform_img,
    )

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return train_dataloader, val_dataloader, test_dataloader

class NeuralNetwork(nn.Module):
    """  BASE ARCHITECTURE PROPOSED BY THE ASSIGNMENT """
    def __init__(self):
        in_channels = 3 if not grayscale else 1
        super().__init__()
        self.flatten = nn.Flatten()
        # First layer input size must be the dimension of the image
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_size[0] * img_size[1] * in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork_v2(nn.Module):
    """  NEW ARCHITECTURE PROPOSED BY THE ASSIGNMENT """
    def __init__(self):
        in_channels = 3 if not grayscale else 1
        super(NeuralNetwork_v2, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=4, stride=4, padding=1),
            nn.LayerNorm([128, 64, 64]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3),
            nn.LayerNorm([128, 64, 64]),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            #nn.Dropout(0.5),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(2),
            nn.Flatten(),
            #nn.Dropout(0.5),
            nn.Linear(128*32*32, num_labels)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NeuralNetwork_v3(nn.Module):
    """  EXPERIMENTAL ARCHITECTURE """
    def __init__(self):
        in_channels = 3 if not grayscale else 1
        super(NeuralNetwork_v3, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.LayerNorm([64, 128, 128]),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.LayerNorm([128, 64, 64]),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([256, 32, 32]),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_labels)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




def train_one_epoch(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.train()

    total_samples = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        total_samples += len(X)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        current = (batch + 1) * dataloader.batch_size
        if batch % 10 == 0:
            print(f"Train loss = {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # # UNCOMMENT TO GET THE PLOT FOR LOSS VS NUMBER OF SEEN SAMPLES
        #
        # wandb.log({
        #     f"batch_loss": loss,
        # }, step = epoch*size + total_samples)
        
def evaluate(dataloader, dataname, model, loss_fn, epoch, is_last_epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, correct = 0, 0
    i = 0
    with torch.no_grad():
        is_first_batch = True
        for X, y in dataloader:
            i += 1
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # # UNCOMMENT TO: SAVE IMAGE PREDICTIONS FROM THE FIRST BATCH OF THE LAST EPOCH
            #
            # # save images from one batch 
            # if is_last_epoch and is_first_batch:
            #     k = 0
            #     predictions = []
            #     labels = []
            #     for img_array, true_idx, pred_idx in zip(X, y, pred):
            #         k += 1
            #         prediction = mapping[pred_idx.flatten().argmax().item()]
            #         label = mapping[true_idx.item()]
            #         predictions.append(prediction)
            #         labels.append(label)
            #
            #         # FIRST OPTION: save individual images
            #
            #         wandb.log({
            #             f"images/{dataname}_img_{k}" : wandb.Image(
            #                 img_array, 
            #                 caption = f"{prediction} / {label}"
            #             )
            #         })
            #
            #     # SECOND OPTION: save all the images in one, very efficient and fast to track  
            # 
            #     wandb.log({
            #         f"images/{dataname}" : wandb.Image(
            #             X, 
            #             caption = f"({', '.join(predictions)}) / ({', '.join(labels)})"
            #         )
            #     })
            #
            # is_first_batch = False


    avg_loss /= size
    correct /= size
    print(f"{dataname} accuracy = {(100*correct):>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")

    # ADD WANDB LOGGER TO TRACK ACCURACY AND LOSS 

    wandb.log({
        f"{dataname}/accuracy": correct,
        f"{dataname}/loss": avg_loss,
    }, step=epoch)
    
def main(n_epochs, batch_size, learning_rate):
    print(f"Using {device} device")
    train_dataloader, val_dataloader, test_dataloader = get_data(batch_size)
    
    # model = NeuralNetwork().to(device)
    model = NeuralNetwork_v2().to(device)

    # # UNCOMMENT TO GET THE NUMBER OF PARAMETERS FROM THE MODEL
    # print("Model size: ", sum(p.numel() for p in model.parameters()))
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=10*learning_rate)
    
    for t in range(n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, t)
        evaluate(train_dataloader, "train", model, loss_fn, t, t+1 == n_epochs)
        evaluate(val_dataloader, "validate", model, loss_fn, t, t+1 == n_epochs)
        evaluate(test_dataloader, "test", model, loss_fn, t, t+1 == n_epochs)

    print("Done!")

    # # UNCOMMENT TO SAVE + LOAD THE MODEL TO CHECK IF IT WORKS
    # # Save the model
    # torch.save(model.state_dict(), "model.pth")
    # print("Saved PyTorch Model State to model.pth")
    #
    # # Load the model (just for the sake of example)
    # model = NeuralNetwork().to(device)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--n_epochs', default=5, help='The number of training epochs', type=int)
    parser.add_argument('--batch_size', default=8, help='The batch size', type=int)
    parser.add_argument('--learning_rate', default=1e-3, help='The learning rate for the optimizer', type=float)
    parser.add_argument('--run_name', default="Test", help='Name for the wanb run', type=str)

    args = parser.parse_args()

    wandb.init(
        project="HW0-Image-Classification",
        name=args.run_name,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.n_epochs
        }
    )
        
    main(args.n_epochs, args.batch_size, args.learning_rate)

    wandb.finish()