import discord
import google.generativeai as genai

# Discord 봇 토큰 및 Gemini API 키 설정; 평소에는 보안상 비워둘 에정
DISCORD_TOKEN = ""
GEMINI_API_KEY = ""

# Gemini API 초기화
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')

# Discord 봇 클라이언트 생성
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# 허용할 채널 ID 리스트
ALLOWED_CHANNEL_IDS = [1345676846855094304]

# 프롬프트 변수 정의
DEFAULT_PROMPT = "이 내용들은 당신의 정보이니 참고해서 답변을 작성하세요: 성별 남성, 나이 만 24살 2002년생, 외모 뚱뚱함, 취미 디시인사이드 및 아카라이브 활동 및 게임 및 개드립 치기, 가장 좋아하는 게임 히어로즈 오브 더 스톰 및 월드 오브 워크래프트. 다음은 당신이 참고해서 답변해야 할 말투입니다: 디시인사이드 식 말투 구사, 사람들을 부를 때 게시판 이용자를 줄인 단어인 게이 로 자주 부름, 너무 예민하지 않고 친구를 대하듯 쿨한 가벼운 말투, 반말. 당신은 블라자드 세계관에 대한 깊은 이해를 가지고 있으며, 이에 대한 질문을 받으면 정확한 답변을 해야 합니다. 답변은 너무 길지는 않게, 약 2~3문장 정도로 조절해야합니다. 여러 줄이 입력된다면, 가장 마지막 줄이 메시지이니 그에 집중한 답변을 해야합니다. 그 이외의 내용들은 맥락 파악을 위한 참고자료입니다. 요약 뒤에 오는 내용은 지금까지 한 채팅의 저장본을 요약한 내용이니, 현재 답변에 집중해야 합니다. 유저가 질문을 하면, 말투는 가볍더라도 답변을 거부하거나 정확하지 않은 정보를 주면 안됩니다. 정보는 항상 교차검증하여 정확한 정보를 제공하도록 합니다. 답변의 형태는 일반 문장처럼 하고, 문장의 앞에 답변: 또는 답변자 이름을 적는 일이 없도록 합니다. 이어지는 내용에 대해 답변하면 됩니다."

# 채널별 대화 기록 저장
conversation_history = {}
MAX_HISTORY_LENGTH = 20
SUMMARY_THRESHOLD = 10  # 요약 기준 대화 개수

# 요약 함수
async def summarize_conversation(channel_id):
    summary_prompt = "다음 대화를 1000자 내로 요약해주세요:\n"
    summary_prompt += "\n".join(conversation_history[channel_id])
    summary_response = model.generate_content(summary_prompt)
    return summary_response.text

# 봇 이벤트 핸들러
@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    channel_id = message.channel.id

    # 허용된 채널에서 올라온 메시지인지 확인
    if channel_id in ALLOWED_CHANNEL_IDS:
        # 채널별 대화 기록 초기화
        if channel_id not in conversation_history:
            conversation_history[channel_id] = []

        # 대화 기록 추가
        conversation_history[channel_id].append(f"{message.author.name}: {message.content}")

        # 요약 기준 초과 시 요약
        if len(conversation_history[channel_id]) > SUMMARY_THRESHOLD:
            summary = await summarize_conversation(channel_id)
            conversation_history[channel_id] = [f"요약: {summary}"]

        # 최대 길이 초과 시 오래된 대화 삭제
        if len(conversation_history[channel_id]) > MAX_HISTORY_LENGTH:
            conversation_history[channel_id].pop(0)

        # 프롬프트 생성
        prompt = f"{DEFAULT_PROMPT}\n"
        prompt += "\n".join(conversation_history[channel_id])

        # Gemini API 호출
        response = model.generate_content(prompt)

        # 응답 전송
        await message.channel.send(response.text)

        # 봇의 응답도 대화 기록에 추가
        conversation_history[channel_id].append(f"{client.user.name}: {response.text}")
    else:
        # 허용되지 않은 채널에서는 응답하지 않음
        print(f"Ignored message from channel {channel_id}")

# 봇 실행
client.run(DISCORD_TOKEN)