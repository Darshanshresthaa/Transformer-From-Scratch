# Inporting all necesssary Libraries

import os
import torch
import torch.nn as nn
import math
import torch.optim as optim


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

batch_size = 12

epochs = 10000


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
            score = score.masked_fill(self.musk_layer[:T,:T] ==0 ,float('-inf'))   #cause block mignnot be same alowasy


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

        self.projection = nn.Linear(embadded_dim, embadded_dim)   #Baically thi is w0 Which is multiplied with concat of all head
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
        

# Feed Forward Neural Network

class feedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.ann = nn.Sequential(nn.Linear(embadding_dim, embadding_dim*5),
                                nn.GELU(),
                                nn.Linear(embadding_dim*5,embadding_dim),
                                )


    def forward(self,x):
        return self.ann(x)
    

# Transformer Decoder Block


class decoder_block(nn.Module):
    def __init__(self):
        super().__init__()

        self.multi_head_attention = multihead_attention(embadding_dim,n_head,block_size,musking=True)
        self.feedForwardNN = feedForwardNN()
        # Now telling the size of matrix before placing a value for normalization
        self.layer_norm_1 =nn.LayerNorm(embadding_dim)  #LayerNorm -> multihead_Attention -> Add (residual)
        self.layer_norm_2 = nn.LayerNorm(embadding_dim) #LayerNorm ,> Fnn -> Add (residual)

    def forward(self,x):
        x = x+self.multi_head_attention(self.layer_norm_1(x))
        x = x+self.feedForwardNN(self.layer_norm_2(x))
        return x
    


class GPT_model(nn.Module):
    def __init__(self):
        super().__init__()
        # B,T,C = x.shape
        self.tokenizer_emb = nn.Embedding(vocab,embadding_dim)
        self.positional_emb = nn.Embedding(block_size,embadding_dim)
        # self.block = nn.Module()

        blocks_list = []

        for i in range(n_layer):
            block = decoder_block()
            blocks_list.append(block)

        self.blocks = nn.Sequential(*blocks_list)
        self.ln_f = nn.LayerNorm(embadding_dim)
        self.lm_head = nn.Linear(embadding_dim, vocab)

    def forward(self, x, targets=None):
        B, T = x.shape

        # token embedding
        tok_emb = self.tokenizer_emb(x) 
        # positional embedding
        pos = torch.arange(T, device=x.device)
        pos_emb = self.positional_emb(pos)
        # combine
        x = tok_emb + pos_emb

        # transformer
        x = self.blocks(x)
        x = self.ln_f(x)

        # output logits
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B*T, vocab)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx
    

    # Now Preparing Data for input and output FOrmat

data = torch.tensor(encoder(text), dtype=torch.long)
def data_prepare():
        

        ix = torch.randint(len(data) - block_size, (batch_size,))  #batch size indicate how many value inside a tensor

        x = []
        y = []

        for i in ix:
            x_seq = data[i:i+block_size]
            x.append(x_seq)
            y_seq = data[i+1:block_size+1]
            y.append(y_seq)


        x = torch.stack(x)
        y = torch.stack(y)
        x= x.to(device)
        y = y.to(device)

        return x,y



model = GPT_model().to(device)
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(),learning_rate)

model.train()
for epoch in range(epochs):
    x,y = data_prepare()
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("step:", epoch+1, "loss:", loss.item())




