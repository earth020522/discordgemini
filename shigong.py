from memory_module import *
import google.generativeai as genai
import threading
import json
import discord

#GEMINI API KEY
with open("config.json", "r") as f:
    config = json.load(f)

API_KEY = config["geminiapi"]

#DISCORD TOKEN
with open("config.json", "r") as f:
    config = json.load(f)

DISCORD_TOKEN = config["discordtoken"]
ALLOWED_CHANNEL_ID = int(config["channelid"])

# 봇 클라이언트 객체 생성
intents = discord.Intents.default()
intents.message_content = True  # 메시지 내용을 읽기 위한 intent 활성화
client = discord.Client(intents=intents)

#프롬프트 내용 입력
BOT_NAME = "봇 이름 여기에 입력하기!"
EDITABLE_PROMPT = """
여기에 프롬프트 입력하기!
"""

#GEMINI MODEL
mainmodel = 'gemini-2.0-flash-thinking-exp-01-21'

#AUTO RESPONSE TIMEOUT(sec)
IDLE_TIMEOUT = 1800

def generate_gemini_response(user_input, history, embeddings):
    """Gemini 모델을 사용하여 응답을 생성합니다."""
    relevant_memories = find_relevant_memory(user_input, history, embeddings)

    simcontext = "\n".join([f"{memory['speaker']}: {memory['text']}" for memory in relevant_memories])

    prompt = f"""
    현 대화와 유사한 이전 대화 내용:
    {simcontext}

    사용자 입력: {user_input}
    
    대답할 시 참고할 중요사항: {EDITABLE_PROMPT}

    이 정보를 참조해서 대화하듯이 대답을 출력해줘.
    """

    model = genai.GenerativeModel(mainmodel)
    response = model.generate_content(prompt)
    return response.text

def generate_auto_response(history, embeddings):
    """자동으로 Gemini 응답을 생성합니다."""
    prompt = f"""
    이전 대화 내용을 바탕으로 자연스럽게 대화를 이어 나가줘.
    """
    model = genai.GenerativeModel(mainmodel)
    response = model.generate_content(prompt)
    return response.text

def main():
    """메인 함수: 대화 시스템을 실행합니다."""

    genai.configure(api_key = API_KEY)
    print("API key activated")
    
    history, embeddings = load_chat_history()
    timer = None

    #일정시간 답 없을시 타임아웃 처리 후 자동 응답
    
    async def send_discord_message(channel_id: int, message: str):
        """특정 채널에 메시지를 보냅니다."""
        channel = client.get_channel(channel_id)
        if channel:
            await channel.send(message)
        else:
            print(f"경고: ID가 {channel_id}인 채널을 찾을 수 없습니다.")

    def timeout_handler():
        """타이머 만료 시 자동 응답을 생성합니다."""
        print("대화가 없어 자동으로 응답을 생성합니다...")
        auto_response = generate_auto_response(history, embeddings)
        asyncio.run(send_discord_message(ALLOWED_CHANNEL_ID, auto_response))
        history, embeddings = add_chat_log("Gemini", auto_response, history, embeddings)
        print("Response:", auto_response)
        start_timer()

    def start_timer():
        """타이머를 시작합니다."""
        nonlocal timer
        timer = threading.Timer(IDLE_TIMEOUT, timeout_handler)
        timer.start()

    def cancel_timer():
        """타이머를 취소합니다."""
        nonlocal timer
        if timer:
            timer.cancel()

    start_timer()
    
    # 봇이 준비되면 실행되는 이벤트
    @client.event
    async def on_ready():
        print(f'{client.user}가 가동함')
    
    #메시지 보낼 시
    @client.event
    async def on_message(message):
        # 봇 자신이 보낸 메시지는 무시
        if message.author == client.user:
            return

        # 사용자가 보낸 메시지에 답변
        if message.channel.id == ALLOWED_CHANNEL_ID:
            print(f'{message.author} (채널: {message.channel}): {message.content}')
            user_name = message.author
            user_input = message.content
            cancel_timer()
          
            #답변 생성
            print("generating response...")
            gemini_response = generate_gemini_response(user_name+": "+user_input, history, embeddings)
            history, embeddings = add_chat_log(user_name, user_input, history, embeddings)
            history, embeddings = add_chat_log(BOT_NAME, gemini_response, history, embeddings)
           
            #답변을 디코 메시지로 보내기
            print("Response:", gemini_response)
            await message.channel.send(gemini_response)
            start_timer()
    
    #봇 구동
    client.run(DISCORD_TOKEN)

if __name__ == "__main__":
    import asyncio
    main()