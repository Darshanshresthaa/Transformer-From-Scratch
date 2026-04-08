# Inporting all necesssary Libraries

import os
import torch
import torch.nn as nn
import math


# Loading a dummy For coding and debugging
with open("input.txt","r") as f:
    text = f.read()


print(text[:1000])   #Displaying oly 1000-1 characters 

text = text.lower()  #lowering all character  so a == lower(A)

char = sorted(list(set(text)))  #store all unique words
print(char)

char_to_int = {}
int_to_char = {}

# Assigning Unique number to The each unique element from text  begining step of encoder
for i,alphabet in enumerate (char):
    char_to_int[alphabet] = i


# Since we  can not understant Numberic value so this method help to convert numberic value back to text
# Decoding text back from numeric
for i,alphabet in enumerate(char):
    int_to_char[i] = alphabet


def encoder(data):
    result = []
    for char in data:
         result.append(char_to_int[char])
    return result
    

def decoder(data):
    result = ""
    for i in data:
        result += int_to_char[i]
    return result


vocab = len(char)  #All unique Charcater present in Data

embadding_dim = 256  # Now converting all numberic char_to_int vlaue to a matrix of  256 dimension

block_size = 64   #how much previous token model can see

n_layer = 5  #Total encoder or Decoder Block (NX)

n_head = 8  #no of self attention Head


class self_attention(nn.Module):
    def __init__(self,embadded_dim,head_size,block_size,musking = True):
        super().__init__()
        self.key = nn.Linear(embadded_dim,head_size,bias=False)  #(batch, sequence_length, embedding_dim)
        self.value = nn.Linear(embadded_dim,head_size,bias=False)
        self.query = nn.Linear(embadded_dim,head_size,bias=False)
        self.head_size = head_size
        self.musking = musking
        
        if self.musking:
            self.register_buffer("musk_layer",torch.tril(torch.ones(block_size, block_size))) #this move direct to the  gpu block and disable this from training

    def forward(self,x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.transpose(-2,-1) 

        score = q@k

        score = score /math.sqrt(self.head_size)  #(batch, sequence_length, sequence length)

        if self.musking:
            score = score.masked_fill(self.musking[:T,:T] ==0 ,float('-inf'))   #cause block mignnot be same alowasy


        score = torch.softmax(score,dim=-1)  

        out = score @ v  #batch,seq_length,head_size


        return out
    
        
            
class multihead_attention(nn.Module):
    def __init__(self, embadded_dim, n_head, block_size, musking=True):
        super().__init__()

        self.n_head = n_head
        self.head_size = embadded_dim // n_head

        self.heads = nn.ModuleList()

        # create heads
        for i in range(n_head):
            self.heads.append(
                self_attention(embadded_dim, self.head_size, block_size, musking)
            )

        self.projection = nn.Linear(embadded_dim, embadded_dim) #You need projection to combine outputs of all heads into one meaningful representation
        # projection create a matriz fo wts containing emb,emb dim where  out put is multiplied to combine outputs togetehr
    def forward(self, x):

        outputs = []

        # run each head
        for h in self.heads:
            outputs.append(h(x))   # store results  h(x) mean applying forward on that

        # concat all heads
        out = torch.cat(outputs, dim=-1)   #before emb = head,n_shize not head*n_size = emb

        # projection
        out = self.projection(out)

        return out         
        
        