from model import CNN
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

input_channel=1
num_classes=10
learning_rate=0.01
batch_size=128
num_epochs=11
train_losses=[]
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train, val = random_split(dataset, [train_size, val_size])

train_loader=DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
val_loader=DataLoader(dataset=val,batch_size=batch_size,shuffle=True)

model=CNN(in_channels=input_channel,num_classes=num_classes).to(device)

loss_func=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    epoch_loss=0.0
    for data,targets in train_loader:
        data=data.to(device=device)
        targets=targets.to(device=device)
        scores=model(data)
        loss=loss_func(scores,targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        epoch_loss+=loss.item()
    epoch_loss/=len(train_loader)
    train_losses.append(epoch_loss)

def check_accuracy(loader,model):
    correct=0
    samples=0
    model.eval()
    with torch.no_grad():
       for x,y in loader:
          x=x.to(device=device)
          y=y.to(device=device)

          scores=model(x)
          _,predictions=scores.max(1)
          correct+=(predictions==y).sum()
          samples+=predictions.size(0)

    acc=(correct/samples)*100
    return acc
torch.save(model.state_dict(), 'trained_model.pth')
t_acc=check_accuracy(train_loader, model)
val_acc=check_accuracy(val_loader,model)

print("the training accuracy is: ",t_acc.item())
print("the validation accuracy is: ",val_acc.item())

plt.plot(range(1,num_epochs+1),train_losses,label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss vs Epochs')
plt.legend()
plt.show()


