import pandas as pd
import numpy as np
import regex as re
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
import sys
import os

# Check filenames and files
args = sys.argv[1:]
if len(args)!=1:
    sys.exit("\nInvalid number of arguments passed\nModel name has to be passed as argument\n")
model=args[0]
file_name='../data/final_test_file.csv'

print(model)
print(os.curdir)
print(os.listdir(os.path.join('./saved_models/')))
if(not os.path.exists(os.path.join(f'./saved_models/{model}'))):
    sys.exit(f"\nCouldn't find the model: {model} under 'src/saved_models' make sure the saved model is available in the saved_model folder or run the training files to generate and save models")

if(not os.path.exists(file_name)):
    sys.exit(
        f"\nCouldn't find the final test file: {file_name} under 'data/' make sure the processed final test file is available in the data folder or run the data_preprocessing file to generate and save the files")

# checking availbility of cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loding final test file and processing
final_test_file = pd.read_csv(file_name)
def convert_str_array(val):
  val=np.array(val[1:-1].split(), dtype=int)
  return val


final_test_file['encoded_titles_combined'] = final_test_file['encoded_titles_combined'].apply(
    convert_str_array)
final_test_tensor = torch.tensor(final_test_file['encoded_titles_combined'])



# Generating outputs with regular LSTM model (Titles are combined and then encoded)
if('siamese'  not in model):
    class LSTM_simple(nn.Module):
        def __init__(self,vocab_size,embed_dimension,lstm_units,hidden_dimension):
            super().__init__()
            self.embeddings=nn.Embedding(vocab_size,embed_dimension)
            self.lstm_layer1= nn.LSTM(embed_dimension,lstm_units,hidden_dimension,bidirectional=True,batch_first=True)
            self.full_layer1 = nn.Linear(2*hidden_dimension,3)

        def forward(self, text):
            text=text.to(device=device)
            embedded_text=self.embeddings(text)
            lstm_out, (ht, ct) = self.lstm_layer1(embedded_text)
            ht=torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)
            ht=self.full_layer1(ht)
            return ht
    simple_LSTM_model= LSTM_simple(49491, 256, 128, 128).to(device=device)
    simple_LSTM_model.load_state_dict(torch.load(f'./saved_models/{model}'))
    simple_LSTM_model.eval()
    with torch.no_grad():
        outputs_predicted=simple_LSTM_model(final_test_tensor)
        # print(outputs_predicted)
        final_test_file['labels']=outputs_predicted.numpy()
        final_test_file.to_csv('../data/LSTM_test_output_predictions_all_columns.csv', index=False)
