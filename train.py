import torch
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
from models.senet import *
from statistics import mean
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch', type=int, default=100)
params = parser.parse_args()

num_epochs = params.epochs
batch_size = params.batch

def my_collate(batch):
    data = [item[0] for item in batch]
    # data = torch.LongTensor(data)
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

data_root = 'train/'
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])
train_dataset = datasets.ImageFolder(root=data_root, transform=base_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = se_resnet18(120).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay = 5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

def run(epoch):
    running_loss = 0.0
    top5_acc,top1_acc = [],[]
    for batch_num, (inputs, labels) in enumerate(train_loader):
        # print('Batch: ',inputs[0],labels)
        model.train()
        inputs,labels = torch.stack(inputs).to(device), labels.to(device)
        # inputs,labels = torch.unsqueeze(inputs[0],0).to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels).to(device)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        model.eval()
        top5 = torch.topk(outputs,k=5)[1]
        top1 = torch.topk(outputs,k=1)[1]
        top5_acc.append(mean([int(label.item() in top5[i]) for i,label in enumerate(labels)]))
        top1_acc.append(mean([int(label.item() in top1[i]) for i,label in enumerate(labels)]))

        if batch_num % 10 == 0:
            print (epoch, batch_num, running_loss, mean(top5_acc))
            running_loss = 0.0

    torch.save(model.state_dict(), f'checkpoint/model.{epoch}')
    gc.collect()
    torch.cuda.empty_cache()

for epoch in range(10):
    run(epoch)
