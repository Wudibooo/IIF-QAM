# 问题分类
class QuestionClassifier(nn.Module):
    def __init__(self,args):
        super(QuestionClassifier, self).__init__()
        self.embedding = nn.Embedding(args.qc_n_vocab, args.embed, padding_idx=args.qc_n_vocab - 1)
        self.lstm = nn.LSTM(args.embed, args.qc_hidden_size, args.qc_num_layers, bidirectional=True,
                            batch_first=True, dropout=args.dropout)
        self.tanh1 = nn.Tanh()
        self.fc_0 = nn.Linear(args.qc_hidden_size2, 2)
        self.w = nn.Parameter(torch.zeros(args.qc_hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc_1 = nn.Linear(args.qc_hidden_size * 2, args.qc_hidden_size2)

    def forward(self, x):
        x, _ = x
      # embedding
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
      # lstm layers
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
      # activateFunc
        M = self.tanh1(H)  # [128, 32, 256] 
      
        aph = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * aph  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc_1(out)
      # 2-classifer quetions
        out = self.fc_0(out)  # [128, 2]
        return out
