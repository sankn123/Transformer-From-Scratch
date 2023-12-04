import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model,vocab_size):

        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.Embeddings=nn.Embedding(self.vocab_size, self.d_model)
    
    def forward(self, x):
        return self.Embeddings(x) * math.sqrt(self.d_model)
    
class PostinalEncoding(nn.Module):
    def __init__(self,seq_length,d_model,dropout):

        super().__init__()
        self.seq_length=seq_length
        self.d_model=d_model
        self.dropout=dropout
        self.Dropout=nn.Dropout(dropout)

        #create a tensor of size (seq_length,d_model)
        pe=torch.zeros(seq_length,d_model)

        #exapnde its dim 0, that is making a tensor of shape (seq_length,1)
        postion=torch.arrange(0,seq_length,dtype=torch.float()).unsqueeze(1)

        #calculating the denominator term
        denom= torch.exp( torch.arrange(0,self.d_model,2).float() * (-1) * math.log( 10000 ) / d_model)

        pe[:,0::2]= torch.sin(postion*denom)
        pe[:,1::2]= torch.cos(postion*denom)

        #considering first dimension ofr batch size
        pe.unsqueeze(0)

        self.register_buffer('pe',pe)


    def forward(self,x):

        x=x+ ( self.pe[:,:x.shape[1],:] ).requires_grad_(False)
        return self.dropout(x)
    

class LayerNorm(nn.Module):

    def __init__(self,d_model):

        super().__init__()
        self.d_model=d_model
        self.layernorm=nn.LayerNorm(self.d_model)

    def forward(self,x):
        return self.layernorm(x)
    

class FeedForwad(nn.Module):

    def __init__(self,d_model,d_ff,dropout):

        super().__init__()
        self.d_model=d_model
        self.dff=d_ff  #dimension of feed forward layer
        self.dropout=dropout

        self.l1=nn.Linear(d_model,d_ff)
        self.l2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):

        # (batch,sq_length,d_model) --> (batch,sq_length,d_ff) --> relu --> dropout --> (batch,sq_length,d_model)
        return  self.l2( self.dropout( torch.relu( self.l1(x) ) ) )  
    

class MultiHeadAttension(nn.Module):

    def __init__(self,d_model,num_heads,dropout):
        super().__init__()
        self.d_model=d_model
        self.dropout=dropout
        self.num_heads=num_heads

        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.attentionlayer=nn.MultiheadAttention(d_model,num_heads,dropout)

    def forward(self,x,mask):

        query=self.w_q(x)
        key=self.w_k(x)
        value=self.w_v(x)

        attn_output, attn_output_weights= self.attentionlayer.forward(query,key,value,mask,self.dropout)

        return attn_output,attn_output_weights

class ResidualConnection(nn.Module):

    def __init___(self,dropout):

        super().__init__()
        self.dropout=dropout
        self.normlayer=LayerNorm()

    def forward(self,x , sublayer):
        return x + self.dropout( sublayer( self.normlayer(x) ) )  

class EncoderBlock(nn.Module):
    def __init__(self, feed_forward : FeedForwad, self_multihead_attention : MultiHeadAttension, dropout : float):

        super().__init__()
        self.feed_forward=feed_forward
        self.self_multihead_attention=self_multihead_attention
        self.dropout=dropout

        self.residuallayers=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,mask ):
        x= self.residuallayers[0](x, lambda x: self.self_multihead_attention(x,mask)[0] ) 
         # self_multihead_attention this will return a tuple,We want just he first output

class Encoder(nn.Module):

    def __init__(self, layers : nn.ModuleList):
        
        super().__init__()
        self.layers=layers
        self.normlayer=LayerNorm()

    def forward(self,x , mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.normlayer(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attension : MultiHeadAttension, cross_attension : MultiHeadAttension, feedforward : FeedForwad , dropout ):

        super().__init__()
        self.self_attension=self_attension
        self.cross_attension=cross_attension
        self.feedforwad=feedforward
        self.dropout=dropout
        self.residuallayers=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):  # src_mask= source mask that is mask of encoder, tgt_mask=targter mask,that is mask of deocder

        x=self.residuallayers[0](x, lambda x: self.self_attension(x,x,x,tgt_mask))
        x=self.residuallayers[1](x,lambda x: self.cross_attension(x,encoder_output,encoder_output,src_mask)) 
        x=self.residuallayers[2](x, self.feedforwad)

        return x
    
class Decoder(nn.Module):
    
    def __init__(self,layers: nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.normlayer=LayerNorm()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
            
        return self.normlayer(x)
    
# these are all the parts needed to build a  transformer, but one final step is remaning which will convert this to words for this we needd to convert this to embeddings, hence next feed
#forward layer will do this, as this will project our output to the embeddings, this is called projection layer

class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size

        self.proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        # (batch_size,seq_length,d_model) --> (batch_size,seq_length,vocab_size)
        return   torch.log_softmax( self.proj(x), dim=-1 )
    


class Transformer(nn.Module):

    def __init__(self, src_emb: InputEmbeddings , tgt_emb: InputEmbeddings, src_pos: PostinalEncoding, tgt_pos: PostinalEncoding , encoder: Encoder, decoder: Decoder, proj_layer: ProjectionLayer ):
        super().__init__()
        self.src_emb=src_emb
        self.tgt_emb=tgt_emb
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.encoder=encoder
        self.decoder=decoder
        self.proj_layer=proj_layer

    def encode(self,src,src_mask):
        src=self.src_emb(src)
        src=self.src_pos(src)

        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,tgt,src_mask,tgt_mask):

        tgt= self.tgt_emb(tgt)
        tgt=self.tgt_pos(tgt)

        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
        
    def project(self,x):
        return self.proj_layer(x)
    
    
def BuildTrasnformer(src_vocab_szie , src_seq_length , tgt_vocab_size , tgt_seq_length , d_model=512 , d_ff=2048 , dropout=0.1 , num_heads=2 ,N=2):

    #N is number of encoder/decoder blocks

    src_emb=InputEmbeddings(d_model,src_vocab_szie)
    tgt_emb=InputEmbeddings(d_model,tgt_vocab_size)

    src_pos=PostinalEncoding(src_seq_length,d_model,dropout)
    tgt_pos=PostinalEncoding(tgt_seq_length,d_model,dropout)


    # making N encoders
    encoder_blocks=[]
    for _ in range(N):
        feedforwardlayer=FeedForwad(d_model,d_ff,dropout)
        encoder_atten_head=MultiHeadAttension(d_model,num_heads,dropout)
        
        encoder_block=EncoderBlock(feedforwardlayer,encoder_atten_head,dropout)
        encoder_blocks.append(encoder_block)

    # making N decoders
    decoder_blocks=[]
    for _ in range(N):
        feedforwardlayer=FeedForwad(d_model,d_ff,dropout)
        decoder_self_atten=-MultiHeadAttension(d_model,num_heads,dropout)
        decoder_cross_atten=MultiHeadAttension(d_model,num_heads,dropout)

        decoder_block=DecoderBlock(decoder_self_atten,decoder_cross_atten,feedforwardlayer,dropout)
        decoder_blocks.append(decoder_block)

    #creating encoder and deocder
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))

    proj_layer=ProjectionLayer(d_model,tgt_vocab_size)

    transformer=Transformer(src_emb,tgt_emb,src_pos,tgt_pos,encoder,decoder,proj_layer)
    #initialise the transformer
    for p in transformer.parameters():
        if p.dim>1:
            nn.init.xavier_uniform(p)

    
    return transformer


