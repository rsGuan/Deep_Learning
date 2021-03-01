from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch import optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from io import open
import re
import time
import math
import random
import string
import numpy as np
import unicodedata
import pandas as pd
from os import system
from argparse import ArgumentParser
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

#-----setting parameters-----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
Teacher_Forcing_Ratio = 0.5

Epochs = 25000
Print_Per_Time = 100
Plot_Per_Time = 100
Hidden_Size = 512
Dropout_P = 0.1 # Dropout probability --> to decrease the overfitting probability
Learning_Rate = 0.01
MAX_LENGTH = 30

Total_Score = []


#-----Train dataset-----

# Read datasets
Train_Data = pd.read_json('train.json')
Test_Data = pd.read_json('test.json')
New_Test_Data = pd.read_json('new_test.json')
Data_Pairs = [] # (inputs, targets)

# Read data into "Data_Pairs" 
for data, target in zip(Train_Data['input'], Train_Data['target']):
    # Let every data would be read
    num = len(data)
    for i in range(num):Data_Pairs.append([data[i], target])

#-----Data Processing-----

# Transfer the type of word
class Words:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.sos_eos_words = 2  # Count SOS and EOS

    def Letter(self, letter):
        if letter not in self.word2index:
            self.word2index[letter] = self.sos_eos_words
            self.word2count[letter] = 1
            self.index2word[self.sos_eos_words] = letter
            self.sos_eos_words += 1
        else:
            self.word2count[letter] += 1
            
    def Word(self, words):
        for word in words:
            for letter in word:
                self.Letter(letter)

# Distinguish the "input" and "target" in the dataset
Input_Words = Words('input')
Output_Words = Words('target')

# Read a word to train per time
for pair in Data_Pairs:
    Input_Words.Word(pair[0])
    Output_Words.Word(pair[1])

#-----Encoder-----

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size) #LSTM

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden) #LSTM
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#-----Decoder-----

class AttentionDecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, output_size, dropout_p=Dropout_P, max_length=MAX_LENGTH):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attention_weights = F.softmax(
            self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attention_applied = torch.bmm(attention_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attention_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromWords(lang, sentence):
    letters = list(sentence)
    return [lang.word2index[letter] for letter in letters]

def tensorFromWords(lang, sentence):
    indexes = indexesFromWords(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromWords(Input_Words, pair[0])
    target_tensor = tensorFromWords(Output_Words, pair[1])
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < Teacher_Forcing_Ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_per_time = Print_Per_Time, plot_per_time = Plot_Per_Time, learning_rate=Learning_Rate):
    plot_losses = []
    plot_bleu_scores = []
    print_loss_total = 0
    print_bleu_score_total = 0
    plot_loss_total = 0
    plot_bleu_score_total = 0
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(Data_Pairs)) for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        print_loss_total += loss
        plot_loss_total += loss
        print_bleu_score_total += compute_test_score()
        plot_bleu_score_total += compute_test_score()
        Total_Score.append(print_bleu_score_total)

        if iter % print_per_time == 0:
            print_loss_avg = print_loss_total / print_per_time
            print_loss_total = 0
            print_bleu_score_avg = print_bleu_score_total / print_per_time
            print_bleu_score_total = 0
            print('%d/%d - Loss: %.4f - Score: %.4f' % (iter, n_iters, print_loss_avg, print_bleu_score_avg), "Current Max Score: ", round(max(Total_Score)/Print_Per_Time,4))

        if iter % plot_per_time == 0:
            plot_loss_avg = plot_loss_total / plot_per_time
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            plot_bleu_score_avg = plot_bleu_score_total / plot_per_time
            plot_bleu_scores.append(plot_bleu_score_avg)
            plot_bleu_score_total = 0
    Plot('loss', plot_losses)
    Plot('score', plot_bleu_scores)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromWords(Input_Words, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(Output_Words.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def compute_test_score():
    bleu_score = []
    for idx in Test_Data.index:
        input = Test_Data.loc[idx, 'input'][0]
        target = Test_Data.loc[idx, 'target']
        output, attentions = evaluate(encoder1, attention_decoder1, input)
        pred = "".join(output)
        bleu_score.append(compute_bleu(target, pred))
    return sum(bleu_score)/len(bleu_score)
    
#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)
    
def Plot(mode, data):
    plt.plot(data)
    plt.xlabel('Epochs')
    if mode == 'loss':
        plt.title('Training Loss')
        plt.ylabel('CrossEntropy Loss')
    elif mode == 'score':
        plt.title('Bleu-4 Test Score')
        plt.ylabel('Score')

    name = mode+"-512.png"
    plt.savefig(name)
    plt.show()
    
encoder1 = EncoderRNN(Input_Words.sos_eos_words, Hidden_Size).to(device)
attention_decoder1 = AttentionDecoderRNN(Hidden_Size, Output_Words.sos_eos_words, dropout_p=0.1).to(device)
trainIters(encoder1, attention_decoder1, Epochs, print_per_time = Print_Per_Time, plot_per_time = Plot_Per_Time)

print("Max BLEU-4 Score: ", round(max(Total_Score) / Print_Per_Time, 4))
print("----------")

#-----BLEU Score(Test Data)-----

Bleu_Score_Test = []

for idx in Test_Data.index:
    input = Test_Data.loc[idx, 'input'][0]
    target = Test_Data.loc[idx, 'target']
    
    output, attentions = evaluate(encoder1, attention_decoder1, input)
    pred = "".join(output)
    print('='*30)
    print('input:\t', input)
    print('target:\t', target)
    print('pred:\t', pred)
    
    Bleu_Score_Test.append(compute_bleu(target, pred))

print('='*30)
print("Test Data BLEU-4 Score: ", round(sum(Bleu_Score_Test)/len(Bleu_Score_Test),4))
print("----------")

#-----
Bleu_Score_New_Test = []

for idx in New_Test_Data.index:
    input = New_Test_Data.loc[idx, 'input'][0]
    target = New_Test_Data.loc[idx, 'target']
    
    output, attentions = evaluate(encoder1, attention_decoder1, input)
    pred = "".join(output)
    print('='*30)
    print('input:\t', input)
    print('target:\t', target)
    print('prediction:\t', pred)
    
    Bleu_Score_New_Test.append(compute_bleu(target, pred))

print('='*30)
print("New Test Data BLEU-4 Score: ", round(sum(Bleu_Score_New_Test)/len(Bleu_Score_New_Test),4))