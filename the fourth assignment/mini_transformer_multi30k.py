# mini_transformer_multi30k.py
# 小型 Transformer 英德翻译模型
# 用于 Attention Is All You Need 复现 + 2.3 残差连接消融实验
#
# 运行示例：
# python mini_transformer_multi30k.py --data_dir D:\Multi30k_Transformer\multi30k_data --mode baseline
# python mini_transformer_multi30k.py --data_dir D:\Multi30k_Transformer\multi30k_data --mode no_residual

import argparse
import math
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_tokenizer(text: str):
    return text.lower().strip().split()


def build_vocab(texts, min_freq=2, max_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenizer(text))

    stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    itos = list(SPECIAL_TOKENS)

    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if word in stoi:
            continue
        if len(itos) >= max_size:
            break
        stoi[word] = len(itos)
        itos.append(word)

    return stoi, itos


def encode_sentence(text, vocab, add_bos_eos=True):
    ids = []
    if add_bos_eos:
        ids.append(vocab[BOS_TOKEN])
    for tok in simple_tokenizer(text):
        ids.append(vocab.get(tok, vocab[UNK_TOKEN]))
    if add_bos_eos:
        ids.append(vocab[EOS_TOKEN])
    return ids


class TranslationDataset(Dataset):
    def __init__(self, hf_split, src_vocab, tgt_vocab):
        self.data = hf_split
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item["en"]
        tgt_text = item["de"]
        src_ids = encode_sentence(src_text, self.src_vocab, add_bos_eos=True)
        tgt_ids = encode_sentence(tgt_text, self.tgt_vocab, add_bos_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def make_collate_fn(src_pad_idx, tgt_pad_idx):
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
        return src_batch, tgt_batch
    return collate_fn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        attn_out = self.dropout1(attn_out)
        src = self.norm1(src + attn_out) if self.use_residual else self.norm1(attn_out)

        ffn_out = self.dropout2(self.ffn(src))
        src = self.norm2(src + ffn_out) if self.use_residual else self.norm2(ffn_out)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        self_attn_out, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )
        self_attn_out = self.dropout1(self_attn_out)
        tgt = self.norm1(tgt + self_attn_out) if self.use_residual else self.norm1(self_attn_out)

        cross_attn_out, _ = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )
        cross_attn_out = self.dropout2(cross_attn_out)
        tgt = self.norm2(tgt + cross_attn_out) if self.use_residual else self.norm2(cross_attn_out)

        ffn_out = self.dropout3(self.ffn(tgt))
        tgt = self.norm3(tgt + ffn_out) if self.use_residual else self.norm3(ffn_out)
        return tgt


class MiniTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=0.1,
        max_len=256,
        use_residual=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout, use_residual)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout, use_residual)
            for _ in range(num_decoder_layers)
        ])
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_tgt_mask(self, tgt_len, device):
        return torch.triu(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, src, tgt_input):
        device = src.device
        src_key_padding_mask = (src == self.src_pad_idx)
        tgt_key_padding_mask = (tgt_input == self.tgt_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt_input.size(1), device)

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt_input) * math.sqrt(self.d_model)

        memory = self.pos_encoder(src_emb)
        out = self.pos_decoder(tgt_emb)

        for layer in self.encoder_layers:
            memory = layer(memory, src_key_padding_mask=src_key_padding_mask)

        for layer in self.decoder_layers:
            out = layer(
                out,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
        return self.output_layer(out)


def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        non_pad = (tgt_output != model.tgt_pad_idx).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        pred = logits.argmax(dim=-1)
        mask = tgt_output != model.tgt_pad_idx

        correct_tokens += ((pred == tgt_output) & mask).sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()

    return total_loss / max(total_tokens, 1), correct_tokens / max(total_tokens, 1)


@torch.no_grad()
def greedy_translate(model, src_sentence, src_vocab, tgt_vocab, tgt_itos, device, max_len=50):
    model.eval()
    src_ids = encode_sentence(src_sentence, src_vocab, add_bos_eos=True)
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    tgt_ids = [tgt_vocab[BOS_TOKEN]]

    for _ in range(max_len):
        tgt_input = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
        logits = model(src, tgt_input)
        next_id = logits[:, -1, :].argmax(dim=-1).item()
        if next_id == tgt_vocab[EOS_TOKEN]:
            break
        tgt_ids.append(next_id)

    return " ".join([tgt_itos[i] for i in tgt_ids[1:] if i < len(tgt_itos)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"D:\Multi30k_Transformer\multi30k_data")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "no_residual"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--dec_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab_size", type=int, default=10000)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备：{device}")

    raw_dataset = load_from_disk(args.data_dir)
    print(raw_dataset)
    print("样例：", raw_dataset["train"][0])

    print("正在构建词表...")
    src_vocab, src_itos = build_vocab([x["en"] for x in raw_dataset["train"]], args.min_freq, args.max_vocab_size)
    tgt_vocab, tgt_itos = build_vocab([x["de"] for x in raw_dataset["train"]], args.min_freq, args.max_vocab_size)
    print(f"英文词表大小：{len(src_vocab)}")
    print(f"德文词表大小：{len(tgt_vocab)}")

    src_pad_idx = src_vocab[PAD_TOKEN]
    tgt_pad_idx = tgt_vocab[PAD_TOKEN]
    valid_key = "validation" if "validation" in raw_dataset else "valid"

    train_dataset = TranslationDataset(raw_dataset["train"], src_vocab, tgt_vocab)
    valid_dataset = TranslationDataset(raw_dataset[valid_key], src_vocab, tgt_vocab)
    collate_fn = make_collate_fn(src_pad_idx, tgt_pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    use_residual = args.mode == "baseline"
    print("模型模式：", "Baseline，有残差连接" if use_residual else "No-Residual，无残差连接")

    model = MiniTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_residual=use_residual,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量：{num_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    valid_losses = []
    valid_accs = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid Token Acc: {valid_acc * 100:.2f}%")

    save_name = "baseline" if use_residual else "no_residual"
    save_path = Path(f"{save_name}_mini_transformer.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "src_itos": src_itos,
        "tgt_itos": tgt_itos,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "valid_accs": valid_accs,
    }, save_path)
    print(f"模型和日志已保存到：{save_path}")

    print("\n翻译样例：")
    for i in range(3):
        item = raw_dataset[valid_key][i]
        src_text = item["en"]
        gold_text = item["de"]
        pred_text = greedy_translate(model, src_text, src_vocab, tgt_vocab, tgt_itos, device)
        print(f"\nEN:   {src_text}")
        print(f"GOLD: {gold_text}")
        print(f"PRED: {pred_text}")


if __name__ == "__main__":
    main()
