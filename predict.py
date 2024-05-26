# %%
import os
from PIL import Image
import torch, gc
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, Subset
import matplotlib.pyplot as plt
import numpy as np
import random
import streamlit as st
import pandas as pd

# #relaesa VRAM
# gc.collect()
# torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")

# %%

st.title("Phase Shift Prediction")

# load the lookup table lookup_table.csv
lookup_table = pd.read_csv('lookup_table.csv')

# find unique values for the idx column
unique_idx = lookup_table['idx'].unique()

# when the user selects an idx, show the corresponding frequency, duty cycle, bmax, and phase shift
testcase = st.selectbox("Select the test index from the magnet challenge dataset", unique_idx)

#print the freq dc, bmax pshift of the selected idx
freq = lookup_table[lookup_table['idx'] == testcase]['freq'].unique()
dc = lookup_table[lookup_table['idx'] == testcase]['dc'].unique()
bmax = lookup_table[lookup_table['idx'] == testcase]['bmax'].unique()
pshift = lookup_table[lookup_table['idx'] == testcase]['pshift'].unique()

#sort the values
freq.sort()
dc.sort()
bmax.sort()
pshift.sort() 
pshift = pshift - 20

st.write("Frequency:", freq[0], "Duty Cycle:", dc[0], "Bmax:", bmax[0], "Phase Shift:", pshift[0])

testpshift = st.select_slider("Phase Shift", options=pshift, value=pshift[3])

# %%

# list of cases
caseidlist = []

# preprocessing, converting to grayscale, resize to 256 * 256 and normalizing the images, 
# and keep the channel dimension as the first dimension of the tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class CustomDataset(Dataset):
    def __init__(self, folder_name):
        self.image_paths = []  # complete path of all image files
        self.labels = []  # store the label information of all images
        self.case_ids = []  # store the operation condition number corresponding to each image
        
        
        folder_path = os.path.join(folder_name)
        # extract physical properties from the folder name
        _, hmax, hmin, bmax, bmin = folder_name.split('_')
        hmax = float(hmax[4:])
        hmin = float(hmin[4:])
        bmax = float(bmax[4:])
        bmin = float(bmin[4:])
        
        case_id = folder_name.split('_')[0][-3:]  # extract the operation condition number
        # traverse all image files in the folder
        for image_file in sorted(os.listdir(folder_path), key=lambda x: int(x[:-4])):
            # create the complete file path and add it to the list
            self.image_paths.append(os.path.join(folder_path, image_file))
            
            # 解析offset_label
            offset_label = int(image_file[:-4]) - 20

            # add the label information as a tuple to the label list
            self.labels.append((offset_label, hmax, hmin, bmax, bmin))
            self.case_ids.append(case_id)  # store the operation condition number
            caseidlist.append(case_id)
                
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # get the image tensor
        image_path = self.image_paths[idx] # get the image path
        image = Image.open(image_path).convert('L') # convert to grayscale
        image_tensor = transform(image)
    
        # get the label information
        offset_label, hmax, hmin, bmax, bmin = self.labels[idx]

        case_id = self.case_ids[idx]  # get the operation condition number

        # hmax, hmin, bmax, bmin as extra inputs tensor
        extra_inputs = torch.tensor([hmax, hmin, bmax, bmin], dtype=torch.float32)

        # return the image tensor, label tensor, extra inputs tensor, and case_id
        return image_tensor, offset_label, extra_inputs, case_id

# set your dataset folder path
dataset_folder = './data'
# dataset = CustomDataset(dataset_folder)

# %% load data 
# find folder name starts with the testcase
folder_name = [name for name in os.listdir(dataset_folder) if name.startswith(str(testcase))][0]
dataset = CustomDataset(dataset_folder + '/' + folder_name)

# predict the dataset
(image_tensor, offset_label, extra_inputs, case_id) = dataset[testpshift + 20]
image_tensor_unsqueeze = image_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
extra_inputs = extra_inputs.unsqueeze(0).to(device)  # Prepare model input


# %% predict

# define a CNN network 
from nn import CNN

model = CNN().to(device)
model_path_load = './model/model_checkpoint_org_local.pth'
model.load_state_dict(torch.load(model_path_load, map_location=torch.device('cpu')))

predicted_offset = model(image_tensor_unsqueeze, extra_inputs).item()  # Perform prediction
print("true offset: ", offset_label, "predicted offset: ", predicted_offset)

# Extract specific max and min values from extra_inputs
hmax, hmin, bmax, bmin = extra_inputs[0].cpu().numpy()  # Move to CPU and convert to numpy

    # Visualize the image and prediction results
fig, ax = plt.subplots()
plt.title(f"Case: {case_id}, Predicted Time: {predicted_offset:.2f}, Actual Time: {offset_label}\n"
            f"Hmax: {hmax:.4f}, Hmin: {hmin:.4f}, Bmax: {bmax:.4f}, Bmin: {bmin:.4f}")
ax.imshow(image_tensor.squeeze(0).numpy(), cmap='gray')  # Display the image

st.pyplot(fig)

# %%