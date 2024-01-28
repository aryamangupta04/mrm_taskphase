from model import CNN
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch

input_channel=1
num_classes=10
learning_rate=0.001
batch_size=64

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test,batch_size=batch_size,shuffle=True)

model=CNN(in_channels=input_channel,num_classes=num_classes).to(device)

loss_func=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
true_labels=[]
pred_labels=[]
classes=range(0,10)

def check_accuracy(loader,model):
    correct=0
    samples=0
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()
    with torch.no_grad():
       for x,y in loader:
          x=x.to(device=device)
          y=y.to(device=device)

          scores=model(x)
          _,predictions=scores.max(1)
          correct+=(predictions==y).sum()
          samples+=predictions.size(0)
          true_labels.extend(y.cpu().numpy())
          pred_labels.extend(predictions.cpu().numpy())

    return (correct/samples)*100



acc=check_accuracy(test_loader, model)
print("the testing accuracy is: ",acc.item())
f1S=f1_score(true_labels,pred_labels,average='macro')
print("the f1 score is: ",f1S)

conf_matrix = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(classes)
plt.yticks(classes)
thresh = conf_matrix.max() / 2.

for i in range(len(classes)):
    for j in range(len(classes)):
        text = f"{conf_matrix[i, j]:d}\n"
        plt.text(j, i, text, ha="center", va="center", color="white" if conf_matrix[i, j] > thresh else "black")

plt.show()