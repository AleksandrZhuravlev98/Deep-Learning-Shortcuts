
#==============================================================================
# PACKAGES
#==============================================================================

import inspect
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os 

#==============================================================================
# Activate CUDA ===============================================================
#==============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#==============================================================================
# Set the seed  ===============================================================
#==============================================================================

# Set random seed for CPU
torch.manual_seed(123456)
# Set random seed for GPU if available
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123456)

# Set random seed for Python's random module
random.seed(123456)

# Set random seed for NumPy
np.random.seed(123456)


#==============================================================================
# DATA LOADING AND INSPECTION
#==============================================================================

train_dat = pd.read_csv("file_path/train_file.csv")
test_dat = pd.read_csv("file_path/test_file.csv")

print(train_dat.describe())
print(test_dat.describe())
print(train_dat.info())
print(test_dat.info())


TT = train_dat.shape[0]

trainn, validd = train_test_split(train_dat, test_size=0.2)

price_min = train_dat["y_train"].min()

#==============================================================================
# LOSS FUNCTION 
#==============================================================================

def rmsle(actual, predicted):
    return np.sqrt(np.mean(np.square(np.log(actual) - np.log(predicted))))
''



#==============================================================================
#==============================================================================
#==============================================================================
# OUR MODEL
#==============================================================================
#==============================================================================
#==============================================================================


# creating a CNN class
class HousePrices2(nn.Module):
	#  determine what layers and their order in CNN object 
    def __init__(self, price_min):
        super(HousePrices2, self).__init__()
        self.fc1 = nn.Linear(29, 29)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=0.8) #Changed dropout to 0.5!
        
        self.fc2 = nn.Linear(29, 29)
        self.relu2 = nn.ReLU()  # I changed this one 
        self.dropout2 = nn.Dropout(0.8)
        
        self.fc3 = nn.Linear(29, 29)
        self.relu3 = nn.LeakyReLU()
        
        self.dropout3 = nn.Dropout(0.8)
        
        self.fc4 = nn.Linear(29, 29)
        self.relu4 = nn.LeakyReLU()
        
        self.fc5 = nn.Linear(29, 1)
        
        #nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.constant_(self.fc1.bias, 0)
        
        self.price_min = price_min
    
    # progresses data across layers    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        out = self.relu4(out)
        #out = self.dropout3(out)
        
        
        out = self.fc5(out)
        #out = self.relu5(out)
        
        #out = self.fc6(out)
        
        # Cope with outliers 
        out = torch.where(out < 0, torch.tensor(1e-6), out)

        return out

# Define the LOSS function ====================================================

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, actual, predicted):
        #epsilon = 1e-6 
        #predicted = torch.where(predicted < 0, torch.tensor(price_min), predicted)
        return torch.sqrt(torch.mean(torch.square(torch.log(actual) - torch.log(predicted))))

#==============================================================================
# Create Tensors of Data

# Train
X_tensor_train = torch.tensor(trainn.iloc[:, range(0,29)].to_numpy(), dtype=torch.float32)
y_tensor_train = torch.tensor(trainn.iloc[:, 29].to_numpy(), dtype=torch.float32).view(-1, 1)  # Reshape to make it a column vector

#Valid 

X_tensor_valid = torch.tensor(validd.iloc[:, range(0,29)].to_numpy(), dtype=torch.float32)
y_tensor_valid = torch.tensor(validd.iloc[:, 29].to_numpy(), dtype=torch.float32).view(-1, 1)  # Reshape to make it a column vector

#==============================================================================


# set the model to device
model = HousePrices2(price_min=price_min)

# set Loss function with criterion
loss_func = RMSLELoss()

# set learning rate 
lr = 0.0015 # 0.0001

# set optimizer with optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
#total_step = len(dloader)


# train and validate the network===============================================

# Train the model
num_epochs = 4500 # number of iterations

#Store for visualisation 

val_loss_list = []
train_loss_list = [] 

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor_train)
    loss = loss_func(outputs, y_tensor_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #print(loss.item())
        
    # at end of epoch check validation loss and accuracy on validation set
    
    # Calculate validation accuracy
    
    with torch.no_grad():
        model.eval()
        val_outputs = model(X_tensor_valid)
        val_outputs = torch.where(val_outputs < torch.tensor(0), torch.tensor(0), val_outputs)
        val_loss = loss_func(y_tensor_valid, val_outputs)
        val_loss_list.append(val_loss.detach().numpy().flatten()[0])
        train_loss_list.append(loss.detach().numpy().flatten()[0])

    # Print loss and validation loss
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Training Loss: {loss.item():.4f}, "
          f"Validation Loss: {val_loss.item():.4f}")

#==============================================================================
# EVALUATION ==================================================================
#==============================================================================

predictions = model(X_tensor_valid)
actual = y_tensor_valid
predictions = torch.where(predictions < 0, torch.tensor(1e-6), predictions)
torch.sqrt(torch.mean(torch.square(torch.log(actual ) - torch.log(predictions ))))
trial =  torch.sqrt(torch.mean(torch.square(torch.log(actual ) - torch.log(predictions ))))
trial= trial.detach().numpy()

#Visualise 

# Create a list of epochs
epochs = list(range(102, 4501))  # Assuming 15000 epochs

train_loss_list = train_loss_list[101:]
val_loss_list = val_loss_list[101:]

# Create a DataFrame to hold the data

data = pd.DataFrame({'Epoch': epochs, 'Training Loss': train_loss_list, 'Validation Loss': val_loss_list})

# Melt the DataFrame so it's in the right format for Seaborn
melted_data = pd.melt(data, id_vars=['Epoch'], value_vars=['Training Loss', 'Validation Loss'],
                      var_name='Loss Type', value_name='Loss Value')
 
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plot = sns.lineplot(x='Epoch', y='Loss Value', hue='Loss Type', data=melted_data)
plot.set_title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



### Another check - visualise predictions against true values 

true_predicted = pd.DataFrame({"True": validd.iloc[:, 29], 
                               "Pred": predictions.detach().numpy().flatten()})
sns.scatterplot(x=true_predicted["True"], y=true_predicted["Pred"])
sns.histplot(true_predicted["Pred"])

# Plot the 45-degree line through the origin
plt.plot([true_predicted["True"].min(), true_predicted["True"].max()], 
         [true_predicted["True"].min(), true_predicted["True"].max()], 
         linestyle='--', color='gray')

plt.text(0.95, 0.95, f"LR: {lr}\nEpochs: {num_epochs}\nDropout: {dropout_prob}\nTrial: {trial}",
         horizontalalignment='right',
         verticalalignment='top',
         transform=plt.gca().transAxes,  # Use axis coordinates for placement
         bbox=dict(facecolor='white', alpha=0.5))  # Add a semi-transparent white background

plt.show()




#==============================================================================
#=============================== Model Deployment 
#==============================================================================

X_tensor_final = torch.tensor(test_dat.iloc[:, range(0,29)].to_numpy(), dtype=torch.float32)

predictions_final = model(X_tensor_final)
predictions_final= predictions_final.detach().numpy()


os.getcwd() # check in which folder you are working in (the file you write will be in that folder)


surname = "Lupa_Pupa" 
file_name = surname + '.csv'

# write.table(pred_submit, file= file_name, row.names = FALSE, col.names  = FALSE)
pd.DataFrame(predictions_final).to_csv(file_name, index=False, header=False)

# Check your file
checkk =  pd.read_table(file_name)
plt.close("all")

fig, ax = plt.subplots(figsize=(9, 6))
fig.subplots_adjust(left=0.3, right=0.91, top=0.9, bottom=0.1)
plt.plot(checkk)
plt.show()


