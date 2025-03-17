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

#Whisper限25MB、10分鐘
MAX_AUDIO_SIZE = 25 * 1024 * 1024  
MAX_SEGMENT_DURATION = 10 * 60 * 1000  

#存當天日記
user_diary = {}

intents = Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"已啟動{bot.user}")
    try:
        synced = await bot.tree.sync()
        print(f"成功同步{len(synced)}指令")
    except Exception as e:
        print(f"無法同步指令{e}")

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

        #分段避Dc限2000字
        await send_long_message(interaction.channel, final_diary)

        #清當天記錄
        user_diary[user_id] = []
    else:
        await interaction.response.send_message("今天尚未有記錄")

@bot.event
async def on_message(message):
    user_id = message.author.id
    if message.author == bot.user:
        return

    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.endswith(("mp3", "m4a", "wav", "webm", "mp4", "mpeg", "mpga")):
                await message.channel.send("**收到，轉錄中...**")

                audio_path = await download_audio(attachment)

                transcriptions = transcribe_large_audio(audio_path)

                #後處理
                refined_transcriptions = [refine_transcription_with_gpt4(t) for t in transcriptions]

                #存起來+傳送
                if user_id not in user_diary:
                    user_diary[user_id] = []
                user_diary[user_id].extend(refined_transcriptions)

                await send_long_message(message.channel, "**轉錄結果：**\n" + "\n".join(refined_transcriptions))

                os.remove(audio_path)

async def download_audio(attachment):
    response = requests.get(attachment.url)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio.write(response.content)
    temp_audio.close()
    return temp_audio.name

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
        transcription = transcribe_audio_whisper(segment)
        transcriptions.append(transcription)
        os.remove(segment)  # 清暫存

    return transcriptions

def transcribe_audio_whisper(file_path):
    with open(file_path, "rb") as audio:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="text"
        )
    return response.strip()

def refine_transcription_with_gpt4(text):
    #轉錄後處理
    SYSTEM_PROMPT = """
    你是一個專業的逐字稿轉錄助手。請幫助我：
    - 修正語法、分段使內容更通順
    - 添加標點符號（，。！？）
    - 保留原始內容的語氣與意思，不要過度改寫
    - 使用繁體中文
    """

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content.strip()

def generate_daily_summary(full_diary_text):
    #統整一天日記
    SYSTEM_PROMPT = f"""
    Dear ChatGPT，今天是 {get_today_date()}。你是我的日記助手，
    我會在一天中寫下隨機的想法和筆記。在一天結束時，請使用繁體中文幫助我：

    1 整理一天的內容
    - 收集我所有的日記內容，重新組織為流暢且有條理的完整日記
    - 不改變原意，但修正語法，使文章更通順
    - 避免省略細節，但可以適當調整內容的前後順序，使之更合理

    2 摘要&生活洞察
    - 摘要今天日記的關鍵要點
    - 以心理專家/人生導師的角色，給出鼓勵、安慰、分析或建議

    3 生成可行的待辦事項
    """

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
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

async def send_long_message(channel, message):
    for i in range(0, len(message), 2000):
        await channel.send(message[i:i+2000])

bot.run(DISCORD_BOT_TOKEN)
