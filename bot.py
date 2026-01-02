from __future__ import annotations

import asyncio
import io
import logging
import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from src.pipeline import RAGPipeline
from src.pipeline_api import RAGPipelineAPI

# Load environment variables from .env file
load_dotenv()  



LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))
HISTORY_MAX = int(os.getenv("HISTORY_MAX", "30"))
HISTORY_SHOW = 10

LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "35"))

TG_MSG_LIMIT = 3800

# Logging

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_for_law_bot")

# In-memory state

@dataclass
class DialogTurn:
    ts: float
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    rating: Optional[str] = None 
    feedback_text: Optional[str] = None


@dataclass
class UserState:
    user_id: int
    username: str = ""
    first_name: str = ""
    created_at: float = field(default_factory=lambda: time.time())
    last_seen: float = field(default_factory=lambda: time.time())

    selected_section: Optional[str] = None
    selected_point: Optional[str] = None

    history: List[DialogTurn] = field(default_factory=list)

    last_question: Optional[str] = None
    last_answer: Optional[str] = None
    last_sources: List[str] = field(default_factory=list)
    last_sources_visible: bool = True  
    awaiting_feedback_text: bool = False  

USERS: Dict[int, UserState] = {}


def get_user_state(update: Update) -> UserState:
    user = update.effective_user
    assert user is not None
    uid = user.id
    st = USERS.get(uid)
    if not st:
        st = UserState(
            user_id=uid,
            username=user.username or "",
            first_name=user.first_name or "",
        )
        USERS[uid] = st
    st.last_seen = time.time()
    st.username = user.username or st.username
    st.first_name = user.first_name or st.first_name
    return st


def push_history(st: UserState, turn: DialogTurn) -> None:
    st.history.append(turn)
    if len(st.history) > HISTORY_MAX:
        st.history = st.history[-HISTORY_MAX:]

# Markdown helpers

def md_escape(text: str) -> str:
    """Экранирует все спецсимволы MarkdownV2 для обычного текста."""
    if not text:
        return ""
    special = r"_*[]()~`>#+-=|{}.!\\"
    out = []
    for ch in text:
        if ch in special:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def convert_markdown_to_v2(text: str) -> str:
    """
    Конвертирует обычный Markdown (из LLM) в Telegram MarkdownV2.
    Сохраняет форматирование: **bold** → *bold*, __italic__ → _italic_
    """
    import re

    if not text:
        return ""

    # Используем уникальные маркеры, которые точно не встречаются в тексте
    MARKER_START = "\x00FMTSTART\x00"
    MARKER_END = "\x00FMTEND\x00"

    # Список сохраненных форматированных фрагментов
    saved_formats = []

    def save_formatted(match, md_v2_char):
        """Сохраняет отформатированный фрагмент."""
        content = match.group(1)
        # Экранируем содержимое
        escaped = md_escape(content)
        formatted = f"{md_v2_char}{escaped}{md_v2_char}"
        # Сохраняем и возвращаем маркер
        idx = len(saved_formats)
        saved_formats.append(formatted)
        return f"{MARKER_START}{idx}{MARKER_END}"

    # Обрабатываем bold: **text** → *text*
    text = re.sub(r'\*\*(.+?)\*\*', lambda m: save_formatted(m, '*'), text)

    # Обрабатываем italic: __text__ → _text_
    text = re.sub(r'__(.+?)__', lambda m: save_formatted(m, '_'), text)

    # Экранируем весь оставшийся текст
    text = md_escape(text)

    # Возвращаем форматированные фрагменты
    def restore_format(match):
        idx = int(match.group(1))
        return saved_formats[idx]

    text = re.sub(rf"{re.escape(MARKER_START)}(\d+){re.escape(MARKER_END)}",
                  restore_format, text)

    return text


def build_sources_block(sources: List[str]) -> str:
    if not sources:
        return "_Источники не найдены_"
    lines = []
    for i, s in enumerate(sources, 1):
        lines.append(f"{i}\\) {md_escape(s)}")
    return "\n".join(lines)


def split_for_telegram(text: str, limit: int = TG_MSG_LIMIT) -> List[str]:

    if len(text) <= limit:
        return [text]

    parts: List[str] = []
    current: List[str] = []
    cur_len = 0

    for paragraph in text.split("\n"):
        add = paragraph + "\n"
        if cur_len + len(add) > limit and current:
            parts.append("".join(current).rstrip())
            current = [add]
            cur_len = len(add)
        else:
            current.append(add)
            cur_len += len(add)

    if current:
        parts.append("".join(current).rstrip())

    final: List[str] = []
    for p in parts:
        if len(p) <= limit:
            final.append(p)
        else:
            final.extend(textwrap.wrap(p, width=limit, break_long_words=False, replace_whitespace=False))
    return final

# Inline keyboard

def answer_keyboard(sources_visible: bool = True) -> InlineKeyboardMarkup:
    src_btn = "Скрыть источники" if sources_visible else "Показать источники"
    src_cb = "SRC_HIDE" if sources_visible else "SRC_SHOW"

    keyboard = [
        [
            InlineKeyboardButton(src_btn, callback_data=src_cb),
            InlineKeyboardButton("Дай кратко", callback_data="STYLE_SHORT"),
            InlineKeyboardButton("Поясни проще", callback_data="STYLE_SIMPLE"),
        ],
        [
            InlineKeyboardButton("👍", callback_data="RATE_UP"),
            InlineKeyboardButton("👎", callback_data="RATE_DOWN"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)

# Pipeline init 

rag_pipeline: Optional[RAGPipeline] = None


def init_pipeline():
    """
    Инициализирует RAG Pipeline.
    Выбор между локальной моделью и API зависит от переменной USE_API.
    """
    global rag_pipeline
    if rag_pipeline is None:
        use_api = os.getenv("USE_API", "true").lower() == "true"

        if use_api:
            logger.info("Initializing RAG Pipeline with Mistral API...")
            try:
                rag_pipeline = RAGPipelineAPI(
                    pdd_path=os.getenv("PDD_PATH", "data/pdd.json"),
                    cache_dir=os.getenv("FAISS_CACHE_DIR", "faiss_langchain_cache"),
                )
                logger.info("RAG Pipeline (API) ready.")
            except ValueError as e:
                logger.error(f"Ошибка инициализации API: {e}")
                logger.error("Проверьте наличие MISTRAL_API_KEY в .env файле")
                raise
        else:
            logger.info("Initializing RAG Pipeline with local model...")
            rag_pipeline = RAGPipeline(
                pdd_path=os.getenv("PDD_PATH", "data/pdd.json"),
                cache_dir=os.getenv("FAISS_CACHE_DIR", "faiss_langchain_cache"),
            )
            logger.info("RAG Pipeline (local) ready.")

    return rag_pipeline


# Core logic

def format_answer_md(answer: str, sources: List[str], sources_visible: bool) -> str:
    """
    Форматирует ответ для отправки в Telegram с MarkdownV2.
    Сохраняет форматирование из ответа LLM (bold, italic).
    """
    a = convert_markdown_to_v2(answer.strip())
    if sources_visible:
        s = build_sources_block(sources)
        return f"*Ответ:*\n{a}\n\n*Источники:*\n{s}"
    return f"*Ответ:*\n{a}"


def extract_sources_from_context(context_items: List[Dict[str, Any]]) -> List[str]:
    """Извлекает источники из контекста без отображения score."""
    sources = []
    for item in context_items or []:
        src = item.get("source") or "Источник не указан"
        sources.append(str(src))
    return sources


async def run_rag_with_timeout(
    query: str,
    top_k: int,
) -> Tuple[str, List[str], bool]:
   
    pipeline = init_pipeline()

    async def _run() -> Dict[str, Any]:

        return await asyncio.to_thread(
            pipeline.run,
            query,
            top_k_content=top_k,
            include_trace=False,
        )

    try:
        result = await asyncio.wait_for(_run(), timeout=LLM_TIMEOUT_SEC)
        answer = result.get("answer") or "Пустой ответ."
        sources = extract_sources_from_context(result.get("context", []))
        return answer, sources, False
    except Exception as e:
        logger.warning("RAG failed or timed out, fallback to SearchOnly: %s", e)

        try:
            docs = await asyncio.to_thread(pipeline.retriever.search, query, top_k)
            if not docs:
                return "К сожалению, по вашему запросу ничего не найдено.", [], True
            # show top-k snippets
            lines = ["Не успел сгенерировать развернутый ответ, но нашёл релевантные пункты:"]
            sources: List[str] = []
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "Источник не указан")
                snippet = (d.page_content or "").strip().replace("\n", " ")
                snippet = snippet[:240] + ("…" if len(snippet) > 240 else "")
                lines.append(f"{i}. {src}\n   {snippet}")
                sources.append(src)
            return "\n".join(lines), sources, True
        except Exception as e2:
            logger.exception("Fallback SearchOnly failed: %s", e2)
            return "Упс. Система сейчас недоступна (ошибка поиска). Попробуйте позже.", [], True


async def send_markdown_split(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text_md: str,
    keyboard: Optional[InlineKeyboardMarkup] = None,
) -> None:

    chunks = split_for_telegram(text_md, TG_MSG_LIMIT)
    chat_id = update.effective_chat.id  

    for i, chunk in enumerate(chunks):
        await context.bot.send_message(
            chat_id=chat_id,
            text=chunk,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard if i == len(chunks) - 1 else None,
            disable_web_page_preview=True,
        )

# Handlers

START_TEXT = (
    "Привет\\! Я бот *RAG for Law* \\— помогаю искать ответы по ПДД РФ\\.\n\n"
    "*Что умею:*\n"
    "• Трёхуровневый семантический поиск: разделы \\→ пункты \\→ подпункты\n"
    "• Векторный поиск на FAISS для точных совпадений по смыслу\n"
    "• В ответах \\— ссылки на конкретные пункты ПДД\n\n"
    "*Как пользоваться:*\n"
    "• Просто напишите вопрос текстом\n"
    "• Можно открыть источники/скрыть источники кнопкой\n"
    "• Есть /history, /clear, /export\n"
)

HELP_TEXT = (
    "*Команды:*\n"
    "• /start \\— приветствие и инструкция\n"
    "• /history \\— последние 10 вопросов\n"
    "• /clear \\— очистить историю\n"
    "• /export \\— выгрузить историю в txt\n\n"
    "Просто отправьте вопрос \\— я отвечу и покажу источники\\."
)


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = get_user_state(update)
    st.awaiting_feedback_text = False
    await update.message.reply_text(  
        START_TEXT,
        parse_mode=ParseMode.MARKDOWN_V2,
        disable_web_page_preview=True,
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = get_user_state(update)
    st.awaiting_feedback_text = False
    await update.message.reply_text( 
        HELP_TEXT,
        parse_mode=ParseMode.MARKDOWN_V2,
        disable_web_page_preview=True,
    )


async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = get_user_state(update)
    st.awaiting_feedback_text = False

    if not st.history:
        await update.message.reply_text("История пуста.")  
        return

    last = st.history[-HISTORY_SHOW:]
    lines = []
    for i, turn in enumerate(last, 1):
        q = turn.question.strip().replace("\n", " ")
        if len(q) > 180:
            q = q[:180] + "…"
        lines.append(f"{i}. {q}")

    text = "*Последние вопросы:*\n" + "\n".join(md_escape(x) for x in lines)
    await send_markdown_split(update, context, text, keyboard=None)


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = get_user_state(update)
    st.history.clear()
    st.last_question = None
    st.last_answer = None
    st.last_sources = []
    st.last_sources_visible = True
    st.awaiting_feedback_text = False
    await update.message.reply_text("Готово — историю очистил.")  


async def export_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = get_user_state(update)
    st.awaiting_feedback_text = False

    if not st.history:
        await update.message.reply_text("Экспортировать нечего: история пуста.")  
        return

    out = io.StringIO()
    out.write(f"RAG for Law — export for user_id={st.user_id}\n")
    out.write(f"username={st.username} first_name={st.first_name}\n")
    out.write("=" * 60 + "\n\n")

    for t in st.history:
        out.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t.ts)) + "\n")
        out.write("Q: " + t.question.strip() + "\n")
        out.write("A: " + t.answer.strip() + "\n")
        if t.sources:
            out.write("Sources:\n")
            for s in t.sources:
                out.write(f"  - {s}\n")
        if t.rating:
            out.write(f"Rating: {t.rating}\n")
        if t.feedback_text:
            out.write(f"Feedback: {t.feedback_text}\n")
        out.write("\n" + "-" * 60 + "\n\n")

    data = out.getvalue().encode("utf-8")
    bio = io.BytesIO(data)
    bio.name = "rag_for_law_history.txt"

    await update.message.reply_document(  
        document=InputFile(bio),
        caption="История диалогов (txt).",
    )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = get_user_state(update)
    msg = update.message
    if msg is None or msg.text is None:
        return
    text = msg.text.strip()
    if not text:
        return

    if st.awaiting_feedback_text and st.history:
        st.awaiting_feedback_text = False
        st.history[-1].feedback_text = text
        await msg.reply_text("Спасибо! Записал фидбэк")
        return

    scoped_query = text
    if st.selected_section:
        scoped_query = f"[Контекст: Раздел {st.selected_section}] {text}"
    if st.selected_point:
        scoped_query = f"[Контекст: Пункт {st.selected_point}] {scoped_query}"

    await msg.chat.send_action(action="typing")

    answer, sources, used_fallback = await run_rag_with_timeout(
        scoped_query,
        top_k=TOP_K_DEFAULT,
    )

    st.last_question = text
    st.last_answer = answer
    st.last_sources = sources
    st.last_sources_visible = True

    push_history(
        st,
        DialogTurn(
            ts=time.time(),
            question=text,
            answer=answer,
            sources=sources,
        ),
    )

    body = format_answer_md(answer, sources, sources_visible=True)
    await send_markdown_split(update, context, body, keyboard=answer_keyboard(True))


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()

    st = get_user_state(update)
    data = query.data or ""

    if not st.last_answer or not st.last_question:
        await query.edit_message_text("Нет последнего ответа, чтобы выполнить действие.")
        return

    if data == "SRC_HIDE":
        st.last_sources_visible = False
        body = format_answer_md(st.last_answer, st.last_sources, sources_visible=False)
        await query.edit_message_text(
            text=body,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=answer_keyboard(False),
            disable_web_page_preview=True,
        )
        return

    if data == "SRC_SHOW":
        st.last_sources_visible = True
        body = format_answer_md(st.last_answer, st.last_sources, sources_visible=True)
        await query.edit_message_text(
            text=body,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=answer_keyboard(True),
            disable_web_page_preview=True,
        )
        return

    if data == "RATE_UP":
        if st.history:
            st.history[-1].rating = "up"
        await query.answer("Спасибо!")
        return

    if data == "RATE_DOWN":
        if st.history:
            st.history[-1].rating = "down"
        st.awaiting_feedback_text = True
        await query.answer("Ок, понял. Напиши, что было не так — одним сообщением", show_alert=True)
        return

    if data in ("STYLE_SHORT", "STYLE_SIMPLE"):
        style_instruction = (
            "Сократи ответ до 3–5 предложений. Убери лишние детали. "
            "Сохрани точность и ссылки на пункты."
            if data == "STYLE_SHORT"
            else
            "Поясни проще, как для новичка. Без канцелярита. "
            "Но не выдумывай факты, опирайся на ПДД."
        )

        base_q = st.last_question
        if st.selected_section:
            base_q = f"[Контекст: Раздел {st.selected_section}] {base_q}"
        if st.selected_point:
            base_q = f"[Контекст: Пункт {st.selected_point}] {base_q}"

        regen_query = f"{base_q}\n\n[ИНСТРУКЦИЯ К ФОРМАТУ ОТВЕТА]\n{style_instruction}"

        await query.edit_message_text(
            text=md_escape("Переформулирую…"),
            parse_mode=ParseMode.MARKDOWN_V2,
        )

        answer, sources, used_fallback = await run_rag_with_timeout(
            regen_query,
            top_k=TOP_K_DEFAULT,
        )

        st.last_answer = answer
        st.last_sources = sources
        body = format_answer_md(answer, sources, sources_visible=st.last_sources_visible)

        if st.history:
            st.history[-1].answer = answer
            st.history[-1].sources = sources

        await query.edit_message_text(
            text=body,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=answer_keyboard(st.last_sources_visible),
            disable_web_page_preview=True,
        )
        return


    await query.answer("Неизвестное действие.")

# App init / run

def build_app() -> Application:
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("history", history_cmd))
    application.add_handler(CommandHandler("clear", clear_cmd))
    application.add_handler(CommandHandler("export", export_cmd))

    application.add_handler(CallbackQueryHandler(callback_handler))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    return application


def main() -> None:

    try:
        init_pipeline()
    except Exception as e:
        logger.exception("Pipeline warmup failed. Bot will still start: %s", e)

    app = build_app()
    logger.info("Bot started. Polling…")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
