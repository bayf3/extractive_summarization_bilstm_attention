"""
Text preprocessing:
- Sentence splitting
- Cleaning
- Summary sentence alignment with similarity threshold
"""

import os
import json
import glob
import nltk
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch

from multiprocessing import Pool, cpu_count

_SBERT_MODEL = None
nltk.download('punkt')


# RAW_CNN_PATH = "data/raw/cnn/stories/"
# RAW_DM_PATH = "data/raw/dailymail/stories/"
RAW_PATH = "../data/raw_stories_mini/"
OUTPUT_DIR = "../data/labeled_stories_mini/"

def process_single_story(fp):
    """单个文件的完整处理（用于多进程）"""
    article, highlights = parse_story(fp)

    if len(article.strip()) == 0:
        return None

    sentences = split_into_sentences(article)
    if len(sentences) == 0:
        return None

    labels = align_labels(sentences, highlights)

    return {
        "id": os.path.basename(fp),
        "sentences": sentences,
        "labels": labels,
        "highlights": highlights
    }

# --------------------------
# 文本清洗
# --------------------------
def clean_text(text):
    # 去除来源来源标记
    patterns = [
        r"\(CNN\)", r"\[CNN\]", r"cnn", r"CNN",
        r"\(Reuters\)", r"\[Reuters\]",
        r"—", r"--", r"---"
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    # 去除多余空格
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------
# 缩写保护：避免误切分
# --------------------------
def protect_abbreviations(text):
    # 将缩写暂时替换为占位符
    abbr = {
        r"U\.S\.A\.": "USA_PROTECTED",
        r"U\.S\.": "US_PROTECTED",
        r"Mr\.": "Mr_PROTECTED",
        r"Mrs\.": "Mrs_PROTECTED",
        r"Dr\.": "Dr_PROTECTED",
        r"Ms\.": "Ms_PROTECTED",
    }
    for a, token in abbr.items():
        text = re.sub(a, token, text)
    return text


def restore_abbreviations(sentences):
    # 把占位符替换回真实缩写
    restore = {
        "USA_PROTECTED": "U.S.A.",
        "US_PROTECTED": "U.S.",
        "Mr_PROTECTED": "Mr.",
        "Mrs_PROTECTED": "Mrs.",
        "Dr_PROTECTED": "Dr.",
        "Ms_PROTECTED": "Ms.",
    }
    restored = []
    for s in sentences:
        for token, val in restore.items():
            s = s.replace(token, val)
        restored.append(s)
    return restored


# --------------------------
# 解析 .story 文件
# --------------------------
def parse_story(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    article_lines = []
    highlights = []
    is_highlight = False

    for line in lines:
        line = line.strip()

        if line == "@highlight":
            is_highlight = True
            continue

        if is_highlight:
            if len(line) > 0:
                highlights.append(line)
        else:
            if len(line) > 0:
                article_lines.append(line)

    article = clean_text(" ".join(article_lines))
    highlights = [clean_text(h) for h in highlights]

    return article, highlights


# --------------------------
# 句子切分
# --------------------------
def split_into_sentences(text):
    text = protect_abbreviations(text)
    raw = nltk.sent_tokenize(text)
    raw = restore_abbreviations(raw)

    # 去除空句子
    return [s.strip() for s in raw if len(s.strip()) > 1]


# --------------------------
# 标签对齐（相似度 ≥ 0.8）
# --------------------------
def get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer(
            "../../embeddings/all-MiniLM-L6-v2_")
        # _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


# def align_labels(sentences, highlights, 
#                  tfidf_threshold=0.1, 
#                  bert_threshold=0.65,
#                  max_candidates=10):
#     """
#     两阶段筛选标签：
#     第 1 阶段：TF-IDF 快速粗筛，选出可能是摘要句的 candidates
#     第 2 阶段：用 Sentence-BERT 精筛，提高语义匹配质量
#     """

#     if len(sentences) == 0:
#         return []

#     if len(highlights) == 0:
#         return [0] * len(sentences)

#     # -------- Stage 1: TF-IDF 粗筛 --------
#     vectorizer = TfidfVectorizer().fit(sentences + highlights)
#     sent_vecs = vectorizer.transform(sentences)      # (num_sent, dim)
#     high_vecs = vectorizer.transform(highlights)    # (num_highlight, dim)

#     sim = cosine_similarity(sent_vecs, high_vecs)   # (num_sent × num_highlight)
#     max_sim = sim.max(axis=1)

#     # 选出可能是摘要句的候选索引（最多 max_candidates 条）
#     candidates = [i for i, s in enumerate(max_sim) if s >= tfidf_threshold]

#     # 限制候选数量，防止某些长文章太多句子
#     if len(candidates) > max_candidates:
#         # 选相似度最高的前 max_candidates 条
#         top_indices = sorted(range(len(candidates)),
#                              key=lambda i: max_sim[candidates[i]],
#                              reverse=True)[:max_candidates]
#         candidates = [candidates[i] for i in top_indices]

#     # 初始化所有句子为 0
#     labels = [0] * len(sentences)

#     # 如果一个候选句都没有，就全 0
#     if len(candidates) == 0:
#         return labels

#     # -------- Stage 2: Sentence-BERT 精筛 --------

#     model = get_sbert()

#     # 只对候选句编码，提高速度
#     cand_sentences = [sentences[i] for i in candidates]
#     sent_embs = model.encode(cand_sentences, convert_to_tensor=True)
#     high_embs = model.encode(highlights, convert_to_tensor=True)

#     cos = util.cos_sim(sent_embs, high_embs)

#     for idx, sims in zip(candidates, cos):
#         if sims.max().item() >= bert_threshold:
#             labels[idx] = 1

#     return labels

def align_labels(sentences, highlights, 
                 tfidf_threshold=0.1, 
                 bert_threshold=0.65,
                 max_candidates=10,
                 min_summary_sent=1,         # 至少保证1条摘要句
                 fallback_topk=1             # 若全0，选topk句子作为摘要句
                 ):
    """
    两阶段筛选标签（增强版，避免全0标签导致训练无法进行）
    """

    if len(sentences) == 0:
        return []

    if len(highlights) == 0:
        return [0] * len(sentences)

    # -------- Stage 1: TF-IDF 粗筛 --------
    vectorizer = TfidfVectorizer().fit(sentences + highlights)
    sent_vecs = vectorizer.transform(sentences)
    high_vecs = vectorizer.transform(highlights)

    sim = cosine_similarity(sent_vecs, high_vecs)
    max_sim = sim.max(axis=1)

    candidates = [i for i, s in enumerate(max_sim) if s >= tfidf_threshold]

    # 限制候选数量
    if len(candidates) > max_candidates:
        top_indices = sorted(range(len(candidates)),
                             key=lambda i: max_sim[candidates[i]],
                             reverse=True)[:max_candidates]
        candidates = [candidates[i] for i in top_indices]

    labels = [0] * len(sentences)

    if len(candidates) < len(highlights):
        # 直接 fallback（无 TF-IDF 通过者）
        return _fallback_select(sentences, highlights, min_summary_sent, fallback_topk)

    # -------- Stage 2: Sentence-BERT 精筛 --------
    model = get_sbert()
    cand_sentences = [sentences[i] for i in candidates]

    sent_embs = model.encode(cand_sentences, convert_to_tensor=True)
    high_embs = model.encode(highlights, convert_to_tensor=True)

    cos = util.cos_sim(sent_embs, high_embs)  # [num_cand, num_highs]

    # 第一轮标注（固定阈值）
    for idx, sims in zip(candidates, cos):
        if sims.max().item() >= bert_threshold:
            labels[idx] = 1

    # -------- 动态阈值 + fallback --------
    if sum(labels) < len(highlights):
        # Step 1：降低阈值再试一次（乘以 0.8）
        new_bert_th = bert_threshold * 0.8
        for j, (idx, sims) in enumerate(zip(candidates, cos)):
            if sims.max().item() >= new_bert_th:
                labels[idx] = 1

    if sum(labels) < len(highlights):
        # Step 2：fallback（Top-K 最大相似句）
        return _fallback_select(sentences, highlights, min_summary_sent, fallback_topk)

    return labels


def _fallback_select(sentences, highlights, min_summary_sent=1, topk=1):
    """
    当两阶段都失败时，基于 Sentence-BERT 选择 TopK 最相似句作为摘要句
    """
    model = get_sbert()
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    high_embs = model.encode(highlights, convert_to_tensor=True)

    cos = util.cos_sim(sent_embs, high_embs)  # [num_sent, num_highs]
    best_sims = cos.max(dim=1).values.cpu().numpy()  # 每个句子与最佳 highlight 的相似度

    # 选择前 topk 个句子作为摘要句
    top_indices = best_sims.argsort()[::-1][:max(topk, min_summary_sent)]

    labels = [0] * len(sentences)
    for idx in top_indices:
        labels[idx] = 1

    return labels



# --------------------------
# 加载所有 story 文件
# --------------------------
# def load_all_stories(paths):
#     files = []
#     for p in paths:
#         files.extend(glob.glob(p + "*.story"))

#     dataset = []

#     for fp in tqdm(files, desc="Processing stories"):
#         article, highlights = parse_story(fp)

#         # 跳过无正文的 story
#         if len(article.strip()) == 0:
#             continue

#         sentences = split_into_sentences(article)
#         if len(sentences) == 0:
#             continue

#         labels = align_labels(sentences, highlights)

#         dataset.append({
#             "id": os.path.basename(fp),
#             "sentences": sentences,
#             "labels": labels,
#             "highlights": highlights
#         })

#     return dataset

def load_all_stories(paths, num_workers=None):
    """多进程加载 story"""

    files = []
    for p in paths:
        files.extend(glob.glob(p + "*.story"))

    print(f"Total files: {len(files)}")

    # 默认使用所有 CPU 核心
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"Using {num_workers} workers")

    dataset = []

    # 使用 multiprocessing.Pool 并行处理
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(process_single_story, files), total=len(files)):
            if result is not None:
                dataset.append(result)

    return dataset


# --------------------------
# 保存 JSON
# --------------------------
def save_json(data, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------
# 主入口
# --------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading raw stories with multiprocessing...")
    data = load_all_stories(
        [RAW_PATH],
        num_workers=4   # 可改成你 CPU 核心数
    )

    print(f"Total usable samples = {len(data)}")

    # 划分 train / val / test
    train_val, test = train_test_split(data, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    save_json(train, OUTPUT_DIR + "train.json")
    save_json(val, OUTPUT_DIR + "val.json")
    save_json(test, OUTPUT_DIR + "test.json")

    print("Preprocessing completed (multiprocessing)!")


if __name__ == "__main__":
    main()
