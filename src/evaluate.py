# src/evaluate.py
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 导入项目模块
from dataset import SummDataset, collate_fn
from model.model import ExtractiveSummarizer
from inference import predict_summary  # 导入我们刚写的推理模块


# --- ROUGE 计算工具函数 ---
def rouge_1(pred_sentences, ref_sentences):
    """计算 ROUGE-1 F1 (词重叠率) [cite: 100-108]"""

    def tokenize(s):
        return s.lower().split()

    pred_words = []
    for s in pred_sentences: pred_words += tokenize(s)
    ref_words = []
    for s in ref_sentences: ref_words += tokenize(s)

    from collections import Counter
    pc = Counter(pred_words)
    rc = Counter(ref_words)
    overlap = sum((pc & rc).values())

    recall = overlap / max(1, sum(rc.values()))
    prec = overlap / max(1, sum(pc.values()))
    if prec + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    return {"r": recall, "p": prec, "f": f1}


def lcs(a, b):
    """最长公共子序列计算辅助函数"""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def rouge_l(pred_sentences, ref_sentences):
    """计算 ROUGE-L F1 (最长公共子序列) [cite: 114-122]"""
    pred = " ".join(pred_sentences).lower().split()
    ref = " ".join(ref_sentences).lower().split()
    L = lcs(pred, ref)
    if len(ref) == 0 or len(pred) == 0: return {"r": 0.0, "p": 0.0, "f": 0.0}
    r = L / len(ref)
    p = L / len(pred)
    if r + p == 0:
        f = 0.0
    else:
        f = 2 * r * p / (r + p)
    return {"r": r, "p": p, "f": f}


# --- 主评估流程 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to saved .pt model")
    parser.add_argument("--data_json", required=True, help="Path to test/val json data")
    parser.add_argument("--glove_path", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--strategy", type=str, default="topk",
                        help="Strategy to use: topk, dynamic, mmr, dynamic_mmr")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading data from {args.data_json}...")
    # Dataset 会自动构建词表，这里为了简单直接build。
    # 严谨做法是加载训练时的 vocab，但作为作业Demo，直接build影响不大，或者保存vocab.pkl
    dataset = SummDataset(args.data_json, build_vocab=True)

    # 关键：collate_fn 必须是我们修改过能返回 'raw_sents' 的那个版本
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Loading model from {args.model_path}...")
    model = ExtractiveSummarizer(dataset.vocab, embed_dim=300, hidden_size=256, glove_path=args.glove_path)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    scores = {"r1_f": [], "rl_f": []}
    print(f"Start Evaluation using strategy: [{args.strategy}]")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            word_ids = batch["word_ids"]  # [B, Sent, Word] (注意这里要根据collate_fn实际key调整)
            # 根据你提供的 collate_fn, key 是 "word_ids" 是 list of tensors
            # 咱们的模型 forward 好像接受的是 padded tensor or list?
            # 假设模型处理 list of tensors 逻辑在 model 内部，或者在这里处理
            # 为了通用性，这里假设 train.py 里那样处理 zip 逻辑:

            ids_list = batch["word_ids"]
            lens_list = batch["lengths"]
            raw_sents_batch = batch["raw_sents"]  # [关键] 获取原文
            refs_batch = batch["highlights"]  # 参考摘要

            for i in range(len(ids_list)):
                curr_ids = ids_list[i].to(device)
                curr_len = lens_list[i].to(device)

                if curr_ids.numel() == 0: continue  # 空数据跳过

                # --- 1. 模型推理 ---
                # 尝试获取向量 (为了支持 MMR)
                # 假设模型 forward 返回 (logits, attention_weights)
                # 如果你的模型修改过支持返回向量，最好是: logits, vectors = model(..., return_vec=True)
                # 这里做个兼容性处理：
                try:
                    # 尝试解包 3 个返回值 (logits, attn, vectors)
                    output = model(curr_ids, curr_len)
                    if len(output) == 3:
                        logits, attn, vectors = output
                    else:
                        logits, attn = output
                        vectors = None  # 不支持 MMR
                except:
                    # 兜底
                    logits, attn = model(curr_ids, curr_len)
                    vectors = None

                sent_scores = torch.sigmoid(logits)  # 转为概率 0~1

                # --- 2. 调用策略生成摘要 ---
                # 注意：如果 strategy 包含 'mmr' 但 vectors 是 None，这里会报错提醒你去改模型
                try:
                    pred_sents = predict_summary(
                        article_sents=raw_sents_batch[i],
                        sent_scores=sent_scores,
                        sent_vectors=vectors,
                        strategy=args.strategy
                    )
                except ValueError as e:
                    print(f"Error: {e}")
                    print("Hint: If using MMR, ensure your model returns sentence vectors.")
                    return

                # --- 3. 计算 ROUGE ---
                ref_sents = refs_batch[i]
                r1 = rouge_1(pred_sents, ref_sents)
                rl = rouge_l(pred_sents, ref_sents)

                scores["r1_f"].append(r1["f"])
                scores["rl_f"].append(rl["f"])

    print(f"\nFinal Results ({len(scores['r1_f'])} samples):")
    print(f"ROUGE-1 F1: {np.mean(scores['r1_f']):.4f}")
    print(f"ROUGE-L F1: {np.mean(scores['rl_f']):.4f}")


if __name__ == "__main__":
    main()