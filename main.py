import json
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os

app = Flask(__name__)

# 初始化LINE配置
configuration = Configuration(access_token='DE/sKCDx2xsgfxMRUFlpnhVlCYEzwsziptdIVp77wioKFnkIRb79HogrCofCxBqByUxbcI/2dhtrUNLk2H/vF1UAvtYIWatBuJgJpGE3nqf19YHrgz3lGxL5GeVXjykAjUXT8quiNL74LLJc0WRPSQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('da924494fdbebed588a439dade56b76f')

# 讀取json
with open("openai_token.json", "r") as file:
    data = json.load(file)

# 初始化问答机器人
os.environ["OPENAI_API_KEY"] = data['OPENAI_API_KEY']

# 加载文档并创建向量存储
#loader = Docx2txtLoader("C:/Users/11020984/Desktop/LLM/hr_test/M-W-2-001_懲處細則與金額規範-2024.03.01.docx")
#loader = Docx2txtLoader("M-W-2-001_懲處細則與金額規範-2024.03.01.docx")
#splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#texts = loader.load_and_split(splitter)

#embeddings = OpenAIEmbeddings()
#vectorstore = Chroma.from_documents(texts, embeddings)

#loader = Docx2txtLoader("C:/Users/11020984/Desktop/LLM/hr_test/M-W-2-012_服裝儀容規範-2022.12.12.docx")
#loader = Docx2txtLoader("M-W-2-012_服裝儀容規範-2022.12.12.docx")
#pages_new = loader.load_and_split(splitter)
#_ = vectorstore.add_documents(pages_new)

# 文件路径列表
file_paths = [
    "M-W-2-001_懲處細則與金額規範-2024.03.01.docx",
    "M-W-2-012_服裝儀容規範-2022.12.12.docx"
    # 可以添加更多文件路径
]

# 建立向量数据库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# 预处理和存储所有文档
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
for file_path in file_paths:
    loader = Docx2txtLoader(file_path)
    texts = loader.load_and_split(splitter)
    vectorstore.add_documents(texts)

qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3), vectorstore.as_retriever())
chat_history = []

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_message = event.message.text
    global chat_history

    # 查找相关文档
    results = vectorstore.similarity_search(user_message, k=1)  # k=1 只取最相关的文档
    if results:
        relevant_doc = results[0].page_content
        # 使用问答机器人回答问题
        result = qa({"question": user_message + ' (請用繁体中文回答)', "chat_history": chat_history, "context": relevant_doc})
        bot_response = result['answer']
        chat_history.append((user_message, bot_response))
    else:
        bot_response = '沒有相關文檔'

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=bot_response)]
            )
        )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

