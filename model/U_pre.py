from transformers import BertModel, BertConfig
import torch
import torch.nn as nn

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1D, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.dec4 = self.conv_block(512, 256)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = nn.Conv1d(64, out_channels, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)

        self.upconv4 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=8, padding="same"),
            nn.ReLU()
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        dec4 = self.upconv4(enc4)
        dec4 = dec4 + enc3  
        dec3 = self.upconv3(dec4)
        dec3 = dec3 + enc2 
        dec2 = self.upconv2(dec3)
        dec2 = dec2 + enc1  
        dec1 = self.dec1(dec2)

        return dec1

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        self.encoder = UNet1D(configs.feature_dim, configs.feature_dim)
        self.projector = nn.Linear(configs.pred_len, configs.pred_len, bias=True)

        transformer_config = BertConfig(
            hidden_size=configs.pred_len,  
            num_attention_heads=4,           
            intermediate_size=configs.pred_len * 4,  
            hidden_dropout_prob=0.1,         
            attention_probs_dropout_prob=0.1,  
            num_hidden_layers=1             
        )
        self.transformer = BertModel(transformer_config)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        x_enc = torch.cat([x_enc, x_mark_enc], 2)  #--> B,L,N
        x_enc = x_enc.permute(0, 2, 1) # --> B,N,L

        enc_out = self.encoder(x_enc)  # --> B,N,L

        transformer_input = torch.cat([x_enc, enc_out], dim=1)  
        transformer_out = self.transformer(
            inputs_embeds=transformer_input
        ).last_hidden_state 

        transformer_out = transformer_out[:, :x_enc.size(1), :]   
        dec_out = self.projector(transformer_out)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
