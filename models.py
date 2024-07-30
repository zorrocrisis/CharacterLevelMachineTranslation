import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
    ):

        super(Attention, self).__init__()
        "Luong et al. general attention (https://arxiv.org/pdf/1508.04025.pdf)"
        self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        query,
        encoder_outputs,
        src_lengths,
    ):
        # query: (batch_size, max_tgt_len, hidden_dim)
        # encoder_outputs: (batch_size, max_src_len, hidden_dim)
        # src_lengths: (batch_size)
        # we will need to use this mask to assign float("-inf") in the attention scores
        # of the padding tokens (such that the output of the softmax is 0 in those positions)
        # Tip: use torch.masked_fill to do this
        # src_seq_mask: (batch_size, max_src_len)
        # the "~" is the elementwise NOT operator
        src_seq_mask = ~self.sequence_mask(src_lengths)
        
        # src_seq_mask = [64, 1, 19]
        src_seq_mask = src_seq_mask.unsqueeze(1)
        
        # query = [64, 21, 128]
        # encoder_outputs = [64, 19, 128]
        # src_lengths = [64]

        batch_size, max_tgt_len, hidden_dim = query.size()
        
        query = query.reshape(batch_size * max_tgt_len, hidden_dim)
        query = self.linear_in(query)
        query = query.reshape(batch_size, max_tgt_len, hidden_dim)

        # attn_score = [64, 21, 19]
        attn_score = torch.bmm(query, encoder_outputs.transpose(1, 2).contiguous())

        attn_score = torch.masked_fill(attn_score, src_seq_mask, float("-inf"))

        alignment = torch.softmax(attn_score, -1)

        c = torch.bmm(alignment, encoder_outputs)

        # attn_out = [64, 21, 128]
        attn_out = torch.tanh(self.linear_out(torch.cat([c, query], dim=2)))


        # attn_out: (batch_size, max_tgt_len, hidden_size)

        return attn_out

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (
            torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
        )


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        
        # src = [64, 19]

        # embed = [64, 19, 128]
        embed = self.dropout(self.embedding(src))

        packed_embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        
        # final_hidden[0] = hidden = [2, 64, 64]
        # final_hidden[1] = cell = [2, 64, 64]
        packed_output, final_hidden = self.lstm(packed_embed)

        # enc_output = [64, 19, 128], 128
        enc_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        enc_output = self.dropout(enc_output)

        # final_hidden[0] = hidden = [1, 64, 128] -> do we need 2 instead of 1 (nLayers * nDirections or nLayers) 
        # final_hidden[1] = cell = [1, 64, 128] -> do we need 2 instead of 1 (nLayers * nDirections or nLayers) 
        final_hidden = self.reshape_hidden(final_hidden)

        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        
        return enc_output, final_hidden
    
    def reshape_hidden(self, hidden):
        if isinstance(hidden, tuple):
            return tuple(self.merge_tensor(h) for h in hidden)
        else:
            return self.merge_tensor(hidden)

    def merge_tensor(self, state_tensor):
        forward_states = state_tensor[::2]
        backward_states = state_tensor[1::2]
        final_states = torch.cat([forward_states, backward_states], 2)
        
        return final_states


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)

        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn #TODO replace with attn

    def forward(
        self,
        tgt, #input
        dec_state, #hidden
        encoder_outputs, #context
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers * num_directions, batch_size, hidden_size)

        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        
        if(tgt.size(1) > 1):
            tgt_embedded = self.embedding(tgt[: , :-1])
        else:
            tgt_embedded = self.embedding(tgt)

        tgt_embedded = self.dropout(tgt_embedded)

        outputs, dec_state = self.lstm(tgt_embedded, dec_state)
        
        if self.attn is not None:
            outputs = self.attn(outputs, encoder_outputs, src_lengths)
            
        outputs = self.dropout(outputs)

        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)
        
        return outputs, dec_state


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
