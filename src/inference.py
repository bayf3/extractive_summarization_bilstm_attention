# src/inference.py
import torch
import torch.nn.functional as F


def predict_summary(article_sents, sent_scores, sent_vectors=None, strategy="topk"):
    """
    统一推理函数：将模型分数转换为最终摘要文本。
    符合文档要求：
    1. 权重排序 -> 选择 Top-K [cite: 143]
    2. 调整句子顺序 (Re-ordering)
    3. 截断过长句子 (单句<=50词)
    4. 扩展功能：动态 K 值  和 MMR 去重

    Args:
        article_sents (list[str]): 原文句子列表
        sent_scores (Tensor): 模型预测的概率分数 [Num_Sents]
        sent_vectors (Tensor, optional): 句子向量 [Num_Sents, Hidden_Dim] (MMR策略需要)
        strategy (str): 'topk', 'dynamic', 'mmr', 'dynamic_mmr'
    """
    num_sents = len(article_sents)
    if num_sents == 0:
        return []

    # --- 1. 确定 K 值 (固定 vs 动态) ---
    use_dynamic_k = "dynamic" in strategy
    if use_dynamic_k:
        # 文档 5.3 建议的动态 K 值策略
        if num_sents <= 10:
            k = 2
        elif num_sents <= 20:
            k = 3
        else:
            k = 4
        k = min(k, num_sents)
    else:
        k = 3  # 默认固定 K=3

    # --- 2. 句子选择策略 (Top-K vs MMR) ---
    use_mmr = "mmr" in strategy
    selected_indices = []

    if not use_mmr:
        # === 简单 Top-K 策略 ===
        # 选分数最高的 k 个
        if k >= num_sents:
            selected_indices = list(range(num_sents))
        else:
            _, topk_indices = torch.topk(sent_scores, k)
            selected_indices = topk_indices.tolist()

    else:
        # === MMR 去重策略 (扩展方向) ===
        if sent_vectors is None:
            raise ValueError("Using MMR strategy requires sentence vectors, but got None.")

        alpha = 0.7  # 平衡因子：0.7 关注相关性，0.3 关注去重
        candidates = list(range(num_sents))

        while len(selected_indices) < k and candidates:
            mmr_scores = []
            for i in candidates:
                # 相关性 (Relevance)
                relevance = sent_scores[i].item()

                # 冗余度 (Redundancy)
                redundancy = 0.0
                if selected_indices:
                    # 计算当前候选句与已选句集的最大相似度
                    curr_vec = sent_vectors[i].unsqueeze(0)
                    sel_vecs = sent_vectors[selected_indices]
                    # 计算 Cosine Similarity
                    sims = F.cosine_similarity(curr_vec, sel_vecs)
                    redundancy = torch.max(sims).item()

                score = alpha * relevance - (1 - alpha) * redundancy
                mmr_scores.append(score)

            # 选 MMR 分数最高的
            best_idx_in_candidates = mmr_scores.index(max(mmr_scores))
            best_idx = candidates[best_idx_in_candidates]

            selected_indices.append(best_idx)
            candidates.pop(best_idx_in_candidates)

    # --- 3. 顺序恢复 (Re-ordering)  ---
    # 摘要句子必须按照原文出现的顺序排列
    selected_indices.sort()

    # --- 4. 后处理：截断过长句子  ---
    final_sents = []
    for idx in selected_indices:
        sent = article_sents[idx]
        words = sent.split()
        # 文档要求：单句 <= 50 词
        if len(words) > 50:
            sent = " ".join(words[:50])
        final_sents.append(sent)

    return final_sents