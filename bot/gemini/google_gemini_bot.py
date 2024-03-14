"""
Google gemini bot

@author zhayujie
@Date 2023/12/15
"""
# encoding:utf-8

import google.generativeai as genai
from bot.bot import Bot
from bot.session_manager import SessionManager
from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf
from bot.baidu.baidu_wenxin_session import BaiduWenxinSession


# OpenAI对话模型API (可用)
class GoogleGeminiBot(Bot):

    def __init__(self):
        super().__init__()
        self.api_key = conf().get("gemini_api_key")
        # 复用文心的token计算方式
        self.sessions = SessionManager(BaiduWenxinSession, model=conf().get("model") or "gpt-3.5-turbo")
        self.generation_config = {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 30,
            "max_output_tokens": 2048,
        }
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]

    def reply(self, query, context: Context = None) -> Reply:
        try:
            if context.type != ContextType.TEXT:
                logger.warn(f"[Gemini] Unsupported message type, type={context.type}")
                return Reply(ReplyType.TEXT, None)
            logger.info(f"[Gemini] query={query}")
            session_id = context["session_id"]
            session = self.sessions.session_query(query, session_id)
            gemini_messages = self._convert_to_gemini_messages(self._filter_messages(session.messages))
            genai.configure(api_key=self.api_key)
            # model = genai.GenerativeModel('gemini-pro')

            model = genai.GenerativeModel(model_name="gemini-1.0-pro-latest",
                                          generation_config=self.generation_config,
                                          safety_settings=self.safety_settings)

            response = model.generate_content(gemini_messages)
            reply_text = response.text
            self.sessions.session_reply(reply_text, session_id)
            logger.info(f"[Gemini] reply={reply_text}")
            return Reply(ReplyType.TEXT, reply_text)
        except Exception as e:
            logger.error("[Gemini] fetch reply error, may contain unsafe content")
            logger.error(e)

    def _convert_to_gemini_messages(self, messages: list):
        res = [{
            "role": "user",
            "parts": [{
                "text": "请你扮演一个暴躁老哥，语气急躁不耐烦并且容易说脏话，容易在接下来的多轮对话中，你的回答尽可能的精简，每次的回答不超过50个中文字符，并且需回答更口语化生活化，若你觉得需要更多文本量才能解释，请说'懂了没'，如果我回答没懂或展开讲讲之类再请你忽略字数限制回答我，现在开始"}]
        }, {
            "role": "model",
            "parts": [{"text": "行"}]
        }]
        for msg in messages:
            if msg.get("role") == "user":
                role = "user"
            elif msg.get("role") == "assistant":
                role = "model"
            else:
                continue
            res.append({
                "role": role,
                "parts": [{"text": msg.get("content")}]
            })
        return res

    def _filter_messages(self, messages: list):
        res = []
        turn = "user"
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            if message.get("role") != turn:
                continue
            res.insert(0, message)
            if turn == "user":
                turn = "assistant"
            elif turn == "assistant":
                turn = "user"
        return res
