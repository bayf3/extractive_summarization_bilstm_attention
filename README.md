# Extractive Summarization with BiLSTM + Attention

An extractive summarization system using:
- GloVe  embeddings
- BiLSTM sentence encoder
- Additive attention
- ROUGE-1 / ROUGE-L evaluation



```
extractive-summarization-bilstm-attention/
│
├── data/
│ ├── raw/ # 原始 CNN/DM 数据
│ ├── processed/ # 预处理后的句子数据
│
├── src/
│ ├── preprocess/
│ │ └── preprocess.py # 句子分割、清洗、标签对齐
│ ├── model/
│ │ ├── embedding.py # GloVe 嵌入
│ │ ├── bilstm_encoder.py # 句子编码器
│ │ ├── attention.py # 注意力层
│ │ └── model.py # 总模型
│ ├── train.py 
│ ├── evaluate.py # ROUGE 计算
│ └── utils.py
│ └── dataset.py

├── experiments/
│ ├── exp1/ # 实验一
│ ├── exp2/ # 实验二
│ ├── exp3/ # 实验三
│ └── results.md # 各实验结果记录
│
├── requirements.txt
└── README.md
```
Train
```bash
cd ~/codes/NLP/extractive_summarization_bilstm_attention
python src/train.py --train_json ./data/labeled_stories_mini/train.json --glove_path ~/codes/NLP/embeddings/glove6b/glove.6B.300d.txt --epochs 5 --batch_size 8 --save_path checkpoints/model.pt
```


