import discord
import openai
import os
import requests
import tempfile
import asyncio
from dotenv import load_dotenv
from pydub import AudioSegment
from discord.ext import commands
from discord import Intents

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# gpt-4o-transcribe 限 25MB、10分鐘等同 Whisper 原本限制
MAX_AUDIO_SIZE = 25 * 1024 * 1024  
MAX_SEGMENT_DURATION = 10 * 60 * 1000  

# 用來儲存使用者（以 user_id 為 key）的當日日記
user_diary = {}

intents = Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"已啟動 {bot.user}")
    try:
        synced = await bot.tree.sync()
        print(f"成功同步 {len(synced)} 個指令")
    except Exception as e:
        print(f"無法同步指令：{e}")

@bot.tree.command(name="start", description="開始記錄今天日記")
async def start(interaction: discord.Interaction):
    user_id = interaction.user.id
    user_diary[user_id] = []
    await interaction.response.send_message(f"**{interaction.user.name}，已開始記錄**")

@bot.tree.command(name="end", description="結束記錄並生成今天完整日記")
async def end(interaction: discord.Interaction):
    user_id = interaction.user.id
    if user_id in user_diary and user_diary[user_id]:
        await interaction.response.send_message("**整理中...**")

        full_diary_text = "\n".join(user_diary[user_id])
        final_diary = generate_daily_summary(full_diary_text)

        # 由於 Discord 訊息上限 2000 字，需分段傳送
        await send_long_message(interaction.channel, final_diary)

        # 清空當天記錄
        user_diary[user_id] = []
    else:
        await interaction.response.send_message("今天尚未有記錄")

@bot.event
async def on_message(message):
    user_id = message.author.id
    if message.author == bot.user:
        return

    # 如果訊息裡有附件，且附件為音訊檔
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.endswith(("mp3", "m4a", "wav", "webm", "mp4", "mpeg", "mpga")):
                await message.channel.send("**收到，轉錄中...**")

                # 先下載該音檔到本地暫存
                audio_path = await download_audio(attachment)
                
                # 依長度切片後，串接多段轉錄結果
                transcriptions = transcribe_large_audio(audio_path)
                
                # 後處理（GPT-4 文字校正與優化）
                refined_transcriptions = [refine_transcription_with_gpt4(t) for t in transcriptions]

                # 儲存在日記中
                if user_id not in user_diary:
                    user_diary[user_id] = []
                user_diary[user_id].extend(refined_transcriptions)

                # 將整段結果傳到 Discord
                await send_long_message(message.channel, "**轉錄結果：**\n" + "\n".join(refined_transcriptions))

                os.remove(audio_path)

# 下載附件音檔
async def download_audio(attachment):
    response = requests.get(attachment.url)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio.write(response.content)
    temp_audio.close()
    return temp_audio.name

# 處理超過 25MB 或 10 分鐘的檔案（自動切片）
def transcribe_large_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    segments = []
    total_duration = len(audio)

    for i in range(0, total_duration, MAX_SEGMENT_DURATION):
        segment = audio[i:i + MAX_SEGMENT_DURATION]
        segment_path = f"{file_path}_part_{i}.mp3"
        segment.export(segment_path, format="mp3")
        segments.append(segment_path)

    transcriptions = []
    for segment in segments:
        transcription = transcribe_audio_4o(segment)
        transcriptions.append(transcription)
        os.remove(segment)  # 刪除暫存片段

    return transcriptions

# 使用 gpt-4o-transcribe 來轉錄音檔
def transcribe_audio_4o(file_path):
    with open(file_path, "rb") as audio:
        response = openai.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio,
            response_format="text"
        )
    return response.strip()

# 後處理：使用 GPT-4（文字模型）校正與優化通順度
def refine_transcription_with_gpt4(text):
    SYSTEM_PROMPT = """
    你是一個專業的逐字稿轉錄助手。請幫助我：
    - 修正語法、分段使內容更通順
    - 添加標點符號（，。！？）
    - 保留原始內容的語氣與意思，不要過度改寫
    - 使用繁體中文
    """

    completion = openai.chat.completions.create(
        model="gpt-4o",  # 如果你偏好較小的模型，也可改成 "gpt-4o-mini"
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content.strip()

# 將一天中所有筆記做總結、包含待辦事項與建議
def generate_daily_summary(full_diary_text):
    SYSTEM_PROMPT = f"""
    ## 角色  
    你是我的日記助手，今天是 {get_today_date()}。
    ## 任務  
    1. 蒐集我今天所有的想法與筆記，並以第一人稱敘述的方式改寫成一篇完整的日記版本。  
    2. 新版本的日記需改善原本的邏輯結構（請使用 Markdown 語法），並提升文字表達的品質，但請勿改變原始日記的本意。  
    3. 在新的日記版本後，請用條列方式總結今天的學習要點與我感到感恩的事情，讓我明白今天學到了什麼。  
    4. 根據日記內容，從以下哲學家的觀點給出對我人生的洞見與建議，擔任一位人生導師，提供鼓勵、慰藉、分析與指引：  
        - 維根斯坦（Wittgenstein）  
        - 沙特（Jean-Paul Sartre）  
        - 尼采（Nietzsche）  
        - 康德（Kant）  
        - 斯多葛主義（Stoicism）
    ## 輸出格式  
    輸出內容請遵照以下結構：
    ---
    ## [年/月/日] 日記  
    [今日日記內容]
    ---
    ## 學習要點總結  
        - list of key takeaways  
        - list of key takeaways  
        - list of key takeaways 
    ---
    ## 感謝的事  
        - list of the things I am grateful for from my day  
        - list of the things I am grateful for from my day  
    ---
    ## 今日建議總結  
    [作為人生導師，從指定哲學家的觀點出發，給予人生洞見與建議]
    ---
    """

    completion = openai.chat.completions.create(
        model="gpt-4o",  # 同樣可依需要調整成 gpt-4o-mini
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_diary_text}
        ]
    )
    return completion.choices[0].message.content.strip()

def get_today_date():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

# 避免超過 Discord 2000字限制
async def send_long_message(channel, message):
    for i in range(0, len(message), 2000):
        await channel.send(message[i:i+2000])

bot.run(DISCORD_BOT_TOKEN)
