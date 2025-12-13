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
_SBERT_MODEL = None
nltk.download('punkt')

RAW_CNN_PATH = "data/raw/cnn/stories/"
RAW_DM_PATH = "data/raw/dailymail/stories/"
OUTPUT_DIR = "data/processed/"

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
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def align_labels(sentences, highlights,
                      tfidf_threshold=0.05,
                      bert_threshold=0.7,
                      max_candidates=10):
    """
    标签对齐函数：
    1. TF-IDF 粗筛候选句，数量限制 max_candidates
    2. SBERT 精筛，保证每个摘要句匹配到一个候选句
    输出长度 = len(sentences) 的 0/1 列表
    """
    labels = [0] * len(sentences)
    if len(sentences) == 0 or len(highlights) == 0:
        return labels

    # -------- Stage 1: TF-IDF 粗筛 --------
    vectorizer = TfidfVectorizer().fit(sentences + highlights)
    sent_vecs = vectorizer.transform(sentences)
    high_vecs = vectorizer.transform(highlights)
    sim = cosine_similarity(sent_vecs, high_vecs)
    max_sim = sim.max(axis=1)

    # 候选句索引（阈值 + 数量限制）
    candidates = [i for i, s in enumerate(max_sim) if s >= tfidf_threshold]
    if len(candidates) > max_candidates:
        # 按相似度排序，选前 max_candidates 个
        candidates = sorted(candidates, key=lambda i: max_sim[i], reverse=True)[:max_candidates]

    if len(candidates) == 0:
        return labels

    # -------- Stage 2: SBERT 精筛 --------
    model = get_sbert()
    cand_sentences = [sentences[i] for i in candidates]
    sent_embs = model.encode(cand_sentences, convert_to_tensor=True)
    high_embs = model.encode(highlights, convert_to_tensor=True)
    cos_sim_matrix = util.cos_sim(sent_embs, high_embs)  # [num_cand, num_highlight]

    # 对每个摘要句，贪心选相似度最高的候选句
    for j in range(cos_sim_matrix.size(1)):
        sims = cos_sim_matrix[:, j]
        best_idx = torch.argmax(sims).item()
        labels[candidates[best_idx]] = 1

    return labels


# --------------------------
# 加载所有 story 文件
# --------------------------
def load_all_stories(paths, max_stories=None):
    files = []
    for p in paths:
        files.extend(glob.glob(p + "*.story"))

    if max_stories is not None:
        files = files[:max_stories] 

    dataset = []

    for fp in tqdm(files, desc="Processing stories"):
        article, highlights = parse_story(fp)

        # 跳过无正文的 story
        if len(article.strip()) == 0:
            continue

        sentences = split_into_sentences(article)
        if len(sentences) == 0:
            continue

        labels = align_labels(sentences, highlights)

        dataset.append({
            "id": os.path.basename(fp),
            "sentences": sentences,
            "labels": labels,
            "highlights": highlights
        })

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

    print("Loading raw stories...")
    data = load_all_stories([RAW_CNN_PATH, RAW_DM_PATH])
    print(f"Total usable samples = {len(data)}")

    # 划分 train / val / test
    train_val, test = train_test_split(data, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    save_json(train, OUTPUT_DIR + "train.json")
    save_json(val, OUTPUT_DIR + "val.json")
    save_json(test, OUTPUT_DIR + "test.json")

    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
