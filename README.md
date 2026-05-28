# Discord 語音日記機器人

基於 OpenAI Whisper 和 GPT-4o 的 Discord 語音日記機器人。上傳語音訊息後自動轉錄、後處理，並在一天結束時整合成完整日記與待辦事項。

## 功能

- 語音轉文字（OpenAI Whisper API）
- 自動加標點、修正錯別字（GPT-4o 後處理）
- `/start`：開始記錄當日語音日記
- `/end`：整合當天所有語音，生成完整日記、心理分析摘要與 JSON 待辦清單
- 自動分段處理超大音訊檔案

## 技術

- Python 3 / discord.py
- OpenAI Whisper API（語音轉文字）
- OpenAI GPT-4o（後處理與日記生成）

## 快速開始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 DISCORD_BOT_TOKEN 與 OPENAI_API_KEY
python main.py
```

## 環境變數

| 變數 | 說明 |
|------|------|
| `DISCORD_BOT_TOKEN` | Discord Bot Token |
| `OPENAI_API_KEY` | OpenAI API 金鑰 |
