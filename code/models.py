import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)
        self.full_attn = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        attn1 = self.encoder_attn(encoder_out)
        attn2 = self.decoder_attn(decoder_hidden).unsqueeze(1)
        # equation 4 in paper
        e = self.full_attn(F.tanh(attn1 + attn2)).squeeze(2)
        # equation 5 in paper
        alpha = F.softmax(e, dim=1)
        return alpha


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        encoder_dim=512,
        decoder_dim=512,
        attention_dim=512,
        p=0.5,
        hard_attention=False,
    ):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.hard_attention = hard_attention

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.c_0 = nn.Linear(encoder_dim, decoder_dim)
        self.h_0 = nn.Linear(encoder_dim, decoder_dim)

        if not hard_attention:
            self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        # equation 7 in paper
        self.L_h = nn.Linear(decoder_dim, embed_dim)
        self.L_z = nn.Linear(encoder_dim, embed_dim)
        self.L_o = nn.Linear(embed_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        """Initalize LSTM hidden and cell states with mean encoder ouput."""
        # page 6 of paper
        mean_encoder_out = encoder_out.mean(dim=1)
        h = torch.tanh(self.h_0(mean_encoder_out))
        c = torch.tanh(self.c_0(mean_encoder_out))
        return h, c

    def forward(self, encoder_out, captions, caption_lengths):
        """Forward pass for teacher-forcing training."""
        B = encoder_out.size(0)
        L = encoder_out.size(1)

        embeddings = self.dropout(self.embedding(captions))
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = caption_lengths - 1
        max_T = decode_lengths.max().item()

        predictions = torch.zeros(B, max_T, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(B, max_T, L).to(encoder_out.device)

        log_probs = torch.zeros(B, max_T).to(encoder_out.device)

        for t in range(max_T):
            batch_size_T = sum([l > t for l in decode_lengths])

            alpha = self.attention(encoder_out[:batch_size_T], h[:batch_size_T])

            if not self.hard_attention:
                context = (alpha.unsqueeze(2) * encoder_out[:batch_size_T]).sum(dim=1)
                beta = torch.sigmoid(self.f_beta(h[:batch_size_T]))
                context = beta * context

            else:
                if self.training:
                    dist = torch.distributions.Categorical(alpha)
                    sampled_idx = dist.sample()
                    log_probs[:batch_size_T, t] = dist.log_prob(sampled_idx)
                    batch_idx = torch.arange(batch_size_T).to(encoder_out.device)
                    context = encoder_out[batch_idx, sampled_idx, :]

                else:
                    # At inference, pick the most likely location
                    _, sampled_idx = alpha.max(dim=1)
                    batch_idx = torch.arange(batch_size_T).to(encoder_out.device)
                    context = encoder_out[batch_idx, sampled_idx, :]

            lstm_input = torch.cat([embeddings[:batch_size_T, t, :], context], dim=1)
            h, c = self.decode_step(lstm_input, (h[:batch_size_T], c[:batch_size_T]))
            h = self.dropout(h)

            # equation 7 in paper
            E_y = embeddings[:batch_size_T, t, :]
            L_h_h = self.L_h(h)
            L_z_z = self.L_z(context)
            preds = self.L_o(E_y + L_h_h + L_z_z)

            predictions[:batch_size_T, t, :] = preds
            alphas[:batch_size_T, t, :] = alpha

        return predictions, captions, decode_lengths, alphas, log_probs
