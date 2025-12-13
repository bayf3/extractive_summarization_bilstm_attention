# src/preprocess.py
import os
import json
import glob
import nltk
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer

# 确保下载了 punkt 分词器
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# =================配置区域=================
RAW_CNN_PATH = "data/raw/cnn/stories/"
RAW_DM_PATH = "data/raw/dailymail/stories/"
OUTPUT_DIR = "data/processed/"

# 全局 ROUGE 打分器 (ROUGE-1, 2, L)
_ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


# =========================================

# --------------------------
# 文本清洗
# --------------------------
def clean_text(text):
    patterns = [
        r"\(CNN\)", r"\[CNN\]", r"cnn", r"CNN",
        r"\(Reuters\)", r"\[Reuters\]",
        r"—", r"--", r"---"
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------
# 缩写保护：避免误切分
# --------------------------
def protect_abbreviations(text):
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
# 句子切分
# --------------------------
def split_into_sentences(text):
    text = protect_abbreviations(text)
    raw = nltk.sent_tokenize(text)
    raw = restore_abbreviations(raw)
    return [s.strip() for s in raw if len(s.strip()) > 1]


# --------------------------
# 解析 .story 文件
# --------------------------
def parse_story(file_path):
    with open(file_path, "r", encoding="utf8", errors='ignore') as f:
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
            if len(line) > 0: highlights.append(line)
        else:
            if len(line) > 0: article_lines.append(line)

    article = clean_text(" ".join(article_lines))
    highlights = [clean_text(h) for h in highlights]
    return article, highlights


# --------------------------
# [核心修改] 贪婪 ROUGE 标签对齐
# --------------------------
def align_labels(sentences, highlights, max_oracle_sents=3):
    """
    使用 Greedy ROUGE 策略生成 Oracle Labels。
    原理：每次选择能让当前 ROUGE 分数提升最大的一句话，直到达到数量限制或无法提升。
    这是抽取式摘要的"金标准"做法。
    """
    if not sentences or not highlights:
        return [0] * len(sentences)

    # 将摘要列表拼接成参考文本
    abstract_text = " ".join(highlights)

    selected_indices = []
    current_summary_list = []
    best_score = 0.0

    # 循环选择句子
    while len(selected_indices) < max_oracle_sents:
        best_gain = 0.0
        best_idx = -1

        for i, sent in enumerate(sentences):
            if i in selected_indices:
                continue

            # 尝试加入这句话
            trial_summary = " ".join(current_summary_list + [sent])

            # 计算 ROUGE 分数
            scores = _ROUGE_SCORER.score(abstract_text, trial_summary)
            # 综合指标：通常使用 ROUGE-1 + ROUGE-2 的平均或总和
            current_score = scores['rouge1'].fmeasure + scores['rouge2'].fmeasure

            # 看看是否有提升
            if current_score > best_score:
                gain = current_score - best_score
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

        # 如果找到了能提升分数的句子
        if best_idx != -1:
            best_score += best_gain
            selected_indices.append(best_idx)
            current_summary_list.append(sentences[best_idx])
        else:
            # 如果遍历一圈都无法提升分数，提前结束
            break

    # 生成 0/1 标签
    labels = [0] * len(sentences)
    for idx in selected_indices:
        labels[idx] = 1

    return labels


# --------------------------
# 加载与处理流程
# --------------------------
def load_and_process_stories(paths):
    files = []
    for p in paths:
        files.extend(glob.glob(os.path.join(p, "*.story")))

    dataset = []

    # 进度条
    for fp in tqdm(files, desc="Processing stories"):
        try:
            article, highlights = parse_story(fp)

            if len(article.strip()) == 0 or len(highlights) == 0:
                continue

            sentences = split_into_sentences(article)
            if len(sentences) == 0:
                continue

            # 生成高质量标签
            labels = align_labels(sentences, highlights)

            # 如果全是0，或者全是1，通常是异常数据，可以考虑过滤（这里保留以防万一）
            if sum(labels) == 0:
                # 极其罕见的情况，可以做一个保底：选前三句
                labels[:min(3, len(labels))] = [1] * min(3, len(labels))

            dataset.append({
                "id": os.path.basename(fp),
                "sentences": sentences,
                "labels": labels,
                "highlights": highlights
            })
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue

    return dataset


def save_json(data, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading and processing raw stories with Greedy ROUGE...")
    # 检查路径是否存在
    if not os.path.exists(RAW_CNN_PATH) or not os.path.exists(RAW_DM_PATH):
        print("Error: Raw data paths not found. Please check RAW_CNN_PATH and RAW_DM_PATH.")
        return

    data = load_and_process_stories([RAW_CNN_PATH, RAW_DM_PATH])
    print(f"Total usable samples = {len(data)}")

    if len(data) == 0:
        print("No data found!")
        return

    # 划分 train / val / test
    train_val, test = train_test_split(data, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    save_json(train, os.path.join(OUTPUT_DIR, "train.json"))
    save_json(val, os.path.join(OUTPUT_DIR, "val.json"))
    save_json(test, os.path.join(OUTPUT_DIR, "test.json"))

    print("Preprocessing completed! Labels are now optimized for ROUGE.")


if __name__ == "__main__":
    main()