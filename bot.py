import os
import re
import time
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration for Google Gemini API
class Config:
    def __init__(self):
        self.api_key = None
        self.model_name = "models/gemini-2.0-pro-exp"  # Default model
        self.chunk_size = 5000
        self.max_tokens = 8000
        self.model = None
        self.fixed_ocr_model = "models/gemini-2.0-flash-lite"
        self.mode = "mcq"  # Default mode is MCQ
        self.simple_prompt = (
            "From the following text, create a set of simple quiz questions and answers in Hindi with explanations:\n"
            "- Focus on key facts like names, years, and terms.\n"
            "- Format each pair as: **Q:** [Question] | **A:** [Answer] | **E:** [Explanation]\n"
            "- Keep it concise and in Hindi.\n"
            "Text:\n{}"
        )
        self.mcq_prompt = (
            "From the following text, create exam-oriented multiple choice questions (MCQs) in Hindi with explanations:\n"
            "- Cover the entire content comprehensively without adding or removing information.\n"
            "- Focus on key facts like names, years, and terms to make it exam-ready.\n"
            "- Format each MCQ as:\n"
            "  Q: [Question]\n"
            "  A: [Option A]\n"
            "  B: [Option B]\n"
            "  C: [Option C]\n"
            "  D: [Option D]\n"
            "  Correct: [A/B/C/D]\n"
            "  E: [Explanation]\n"
            "- Keep it concise and in Hindi.\n"
            "Text:\n{}"
        )

    def set_api_key(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def set_model(self, model_name):
        self.model_name = model_name
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)

    def set_mode(self, mode):
        self.mode = mode.lower()

    def is_pro_model(self):
        pro_models = [
            "gemini-1.0-pro",
            "gemini-1.5-pro-latest",
            "models/gemini-2.0-pro-exp",
            "models/gemini-2.0-pro-exp-02-05"
        ]
        return self.model_name in pro_models

config = Config()

AVAILABLE_MODELS = [
    {"full": "gemini-1.5-pro-latest", "short": "1.5-pro-latest"},
    {"full": "gemini-1.0-pro", "short": "1.0-pro"},
    {"full": "gemini-1.5-flash", "short": "1.5-flash"},
    {"full": "models/gemini-2.0-flash-exp", "short": "2.0-flash-exp"},
    {"full": "models/gemini-2.0-flash", "short": "2.0-flash"},
    {"full": "models/gemini-2.0-flash-001", "short": "2.0-flash-001"},
    {"full": "models/gemini-2.0-flash-exp-image-generation", "short": "2.0-flash-exp-img"},
    {"full": "models/gemini-2.0-flash-lite-001", "short": "2.0-flash-lite-001"},
    {"full": "models/gemini-2.0-flash-lite", "short": "2.0-flash-lite"},
    {"full": "models/gemini-2.0-flash-lite-preview-02-05", "short": "2.0-flash-lite-p-0205"},
    {"full": "models/gemini-2.0-flash-lite-preview", "short": "2.0-flash-lite-p"},
    {"full": "models/gemini-2.0-pro-exp", "short": "2.0-pro-exp"},
    {"full": "models/gemini-2.0-pro-exp-02-05", "short": "2.0-pro-exp-0205"},
    {"full": "models/gemini-exp-1206", "short": "exp-1206"},
    {"full": "models/gemini-2.0-flash-thinking-exp-01-21", "short": "2.0-flash-think-0121"},
    {"full": "models/gemini-2.0-flash-thinking-exp", "short": "2.0-flash-think-exp"},
    {"full": "models/gemini-2.0-flash-thinking-exp-1219", "short": "2.0-flash-think-1219"}
]

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

def split_text(text, chunk_size=config.chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_questions(chunk, mode):
    if not config.model:
        if not config.api_key:
            return "❌ पहले Gemini API कुंजी सेट करें (/setkey <key>)।"
        config.set_model(config.model_name)
    try:
        prompt = config.mcq_prompt if mode == "mcq" else config.simple_prompt
        response = config.model.generate_content(
            prompt.format(chunk),
            generation_config=genai.types.GenerationConfig(max_output_tokens=config.max_tokens)
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return f"❌ त्रुटि: {e}"

def ocr_image(image_path):
    try:
        ocr_model = genai.GenerativeModel(config.fixed_ocr_model)
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
        response = ocr_model.generate_content(
            ["Extract text from this image:", {"mime_type": "image/jpeg", "data": img_data}]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return f"❌ OCR त्रुटि: {e}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 **हाय!** मुझे एक .txt, .pdf, या छवि (.jpg/.png/फोटो) भेजें, और मैं उससे क्विज़ बना दूंगा।\n\n"
        "📌 **कैसे काम करता है:**\n"
        "- डिफ़ॉल्ट: MCQ मोड (`gemini-2.0-pro-exp` मॉडल)\n"
        "- **Simple Mode**: मैन्युअल प्रॉम्प्ट सेट करें।\n"
        "- **MCQ Mode**: परीक्षा-उन्मुख MCQs पूरे टॉपिक को कवर करेंगे।\n"
        "- `/setmode simple` या `/setmode mcq` से मोड बदलें।\n\n"
        "🔹 **कमांड्स:**\n"
        "- `/start` - बॉट शुरू करें\n"
        "- `/setkey <key>` - Gemini API कुंजी सेट करें\n"
        "- `/listmodels` - उपलब्ध मॉडल देखें और चुनें\n"
        "- `/setmode <simple/mcq>` - क्विज़ मोड सेट करें\n"
        "- `/setprompt <prompt>` - Simple मोड के लिए प्रॉम्प्ट सेट करें\n"
        "- `/help` - सभी कमांड देखें",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 **सभी कमांड्स:**\n\n"
        "- `/start` - बॉट शुरू करें\n"
        "- `/setkey <key>` - Gemini API कुंजी सेट करें\n"
        "- `/listmodels` - उपलब्ध Gemini मॉडल देखें और चुनें\n"
        "- `/setmode <simple/mcq>` - क्विज़ मोड सेट करें\n"
        "- `/setprompt <prompt>` - Simple मोड के लिए कस्टम प्रॉम्प्ट सेट करें\n"
        "- `/help` - यह संदेश दिखाएँ",
        parse_mode="Markdown"
    )

async def set_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ कृपया API कुंजी प्रदान करें। उदाहरण: `/setkey your-api-key`", parse_mode="Markdown")
        return
    api_key = " ".join(context.args)
    config.set_api_key(api_key)
    await update.message.reply_text("✅ Gemini API कुंजी सेट हो गई। अब मॉडल चुनें: `/listmodels` या डिफ़ॉल्ट उपयोग करें।", parse_mode="Markdown")

async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(model["short"], callback_data=f"model_{model['full']}")] for model in AVAILABLE_MODELS]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("📌 **उपलब्ध Gemini मॉडल:**\nकृपया एक मॉडल चुनें:", reply_markup=reply_markup, parse_mode="Markdown")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data.startswith("model_"):
        model_name = query.data[len("model_"):]
        config.set_model(model_name)
        await query.edit_message_text(f"✅ मॉडल `{model_name}` सेट हो गया। अब फाइल या छवि भेजें।", parse_mode="Markdown")

async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or context.args[0].lower() not in ["simple", "mcq"]:
        await update.message.reply_text("❌ कृपया `simple` या `mcq` चुनें। उदाहरण: `/setmode mcq`", parse_mode="Markdown")
        return
    mode = context.args[0].lower()
    config.set_mode(mode)
    await update.message.reply_text(f"✅ मोड `{mode}` सेट हो गया। अब फाइल या छवि भेजें।", parse_mode="Markdown")

async def set_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ कृपया एक प्रॉम्प्ट प्रदान करें। उदाहरण: `/setprompt नया प्रॉम्प्ट`", parse_mode="Markdown")
        return
    prompt_text = " ".join(context.args)
    config.simple_prompt = prompt_text + "\nText:\n{}"
    config.set_mode("simple")
    await update.message.reply_text("✅ Simple मोड के लिए कस्टम प्रॉम्प्ट सेट हो गया। अब फाइल या छवि भेजें।", parse_mode="Markdown")

def parse_mcq(text):
    mcqs = []
    pattern = r"Q: (.+?)\nA: (.+?)\nB: (.+?)\nC: (.+?)\nD: (.+?)\nCorrect: ([ABCD])\nE: (.+?)(?=\nQ:|$)"
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        question = match.group(1).strip()[:256]
        options = [match.group(i).strip()[:100] for i in range(2, 6)]
        correct = {"A": 0, "B": 1, "C": 2, "D": 3}[match.group(6).strip()]
        explanation = match.group(7).strip()[:1024]
        mcqs.append({"question": question, "options": options, "correct": correct, "explanation": explanation})
    return mcqs

def parse_simple(text):
    simples = []
    pattern = r"\*\*Q:\*\* (.+?) \| \*\*A:\*\* (.+?) \| \*\*E:\*\* (.+?)(?=\n\*\*Q:\*\*|$)"
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        explanation = match.group(3).strip()
        simples.append({"question": question, "answer": answer, "explanation": explanation})
    return simples

async def process_text_and_generate_quiz(update: Update, text_content, progress_msg):
    try:
        text_content = clean_text(text_content)
        if not text_content:
            await progress_msg.edit_text("❌ कोई टेक्स्ट नहीं निकाला जा सका।", parse_mode="Markdown")
            return

        chunks = split_text(text_content)
        total_chunks = len(chunks)
        quiz_output = ""
        is_pro_model = config.is_pro_model() and config.mode == "mcq"

        for i, chunk in enumerate(chunks):
            await progress_msg.edit_text(f"🔄 **प्रगति:** चंक {i+1}/{total_chunks} से प्रश्न बन रहे हैं...", parse_mode="Markdown")
            quiz_output += generate_questions(chunk, config.mode) + "\n\n"
            if is_pro_model and i < total_chunks - 1:
                await progress_msg.edit_text(f"⏳ **प्रतीक्षा:** चंक {i+1}/{total_chunks} पूरा हुआ। 30 सेकंड का अंतराल...", parse_mode="Markdown")
                time.sleep(30)

        if config.mode == "mcq":
            mcqs = parse_mcq(quiz_output)
            if not mcqs:
                await progress_msg.edit_text("❌ MCQ पार्स करने में त्रुटि। प्रॉम्प्ट चेक करें।", parse_mode="Markdown")
                return
            await progress_msg.edit_text(f"✅ **MCQ क्विज़ तैयार है!** ({len(mcqs)} सवाल)", parse_mode="Markdown")
            for i, mcq in enumerate(mcqs, 1):
                logger.info(f"Poll {i}: Q: {mcq['question']} | Options: {mcq['options']} | Correct: {mcq['correct']} | E: {mcq['explanation']}")
                logger.info(f"Lengths - Q: {len(mcq['question'])}, Options: {[len(opt) for opt in mcq['options']]}, E: {len(mcq['explanation'])}")

                question = mcq['question'][:256]
                options = [opt[:100] for opt in mcq['options']][:10]
                explanation = mcq['explanation'][:1024]

                if len(question) > 256 or any(len(opt) > 100 for opt in options) or len(explanation) > 1024:
                    logger.error(f"Invalid lengths in question {i}")
                    continue

                try:
                    await update.message.reply_poll(
                        question=f"प्रश्न {i}: {question}",
                        options=options,
                        type="quiz",
                        correct_option_id=mcq['correct'],
                        is_anonymous=False,
                        explanation=explanation,
                        explanation_parse_mode="Markdown"
                    )
                except Exception as e:
                    logger.error(f"Poll {i} error: {e}")
                    # Fallback to text output for any poll error
                    text_output = (
                        f"**प्रश्न {i}:** {mcq['question']}\n"
                        f"**A:** {mcq['options'][0]}\n"
                        f"**B:** {mcq['options'][1]}\n"
                        f"**C:** {mcq['options'][2]}\n"
                        f"**D:** {mcq['options'][3]}\n"
                        f"**सही उत्तर:** {mcq['options'][mcq['correct']]}\n"
                        f"**विवरण:** {mcq['explanation']}\n"
                    )
                    if len(text_output) > 4000:
                        parts = [text_output[i:i + 4000] for i in range(0, len(text_output), 4000)]
                        for part in parts:
                            await update.message.reply_text(part, parse_mode="Markdown")
                    else:
                        await update.message.reply_text(text_output, parse_mode="Markdown")
                    await update.message.reply_text(
                        f"⚠️ प्रश्न {i} को पोल के बजाय टेक्स्ट के रूप में भेजा गया क्योंकि इसमें त्रुटि हुई: {e}",
                        parse_mode="Markdown"
                    )
        else:  # Simple mode
            simples = parse_simple(quiz_output)
            if not simples:
                await progress_msg.edit_text("❌ Simple प्रश्न पार्स करने में त्रुटि। प्रॉम्प्ट चेक करें।", parse_mode="Markdown")
                return
            output_parts = []
            current_part = f"✅ **Simple क्विज़ तैयार है!** ({len(simples)} सवाल)\n\n"
            
            for i, simple in enumerate(simples, 1):
                question_block = (
                    f"**प्रश्न {i}:** {simple['question']}\n"
                    f"**उत्तर:** {simple['answer']}\n"
                    f"**विवरण:** {simple['explanation']}\n\n"
                )
                if len(current_part) + len(question_block) > 4000:
                    output_parts.append(current_part)
                    current_part = question_block
                else:
                    current_part += question_block
            
            if current_part:
                output_parts.append(current_part)
            
            for part in output_parts:
                await update.message.reply_text(part, parse_mode="Markdown")
            await progress_msg.delete()
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        await progress_msg.edit_text(f"❌ प्रोसेसिंग त्रुटि: {e}", parse_mode="Markdown")

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not config.api_key:
        await update.message.reply_text("❌ पहले `/setkey <key>` सेट करें।", parse_mode="Markdown")
        return

    progress_msg = await update.message.reply_text("⏳ **प्रगति:** फाइल डाउनलोड हो रही है...", parse_mode="Markdown")
    text_content = ""

    try:
        if update.message.document:
            file = update.message.document
            file_obj = await context.bot.get_file(file.file_id)
            file_path = await file_obj.download_to_drive()

            if file.mime_type == "text/plain":
                await progress_msg.edit_text("📝 **प्रगति:** .txt फाइल पढ़ी जा रही है...", parse_mode="Markdown")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            elif file.mime_type == "application/pdf":
                await progress_msg.edit_text("📄 **प्रगति:** .pdf फाइल प्रोसेस हो रही है...", parse_mode="Markdown")
                images = convert_from_path(file_path)
                total_pages = len(images)
                for i, image in enumerate(images):
                    image_path = f"temp_page_{i}.jpg"
                    image.save(image_path, "JPEG")
                    await progress_msg.edit_text(f"🖼️ **प्रगति:** पेज {i+1}/{total_pages} का OCR हो रहा है...", parse_mode="Markdown")
                    text_content += ocr_image(image_path) + "\n"
                    os.remove(image_path)
            elif file.mime_type in ["image/jpeg", "image/png"]:
                await progress_msg.edit_text("🖼️ **प्रगति:** छवि का OCR हो रहा है...", parse_mode="Markdown")
                text_content = ocr_image(file_path)
            else:
                await progress_msg.edit_text("❌ कृपया .txt, .pdf, या .jpg/.png फाइल भेजें।", parse_mode="Markdown")
                os.remove(file_path)
                return

            os.remove(file_path)
            await process_text_and_generate_quiz(update, text_content, progress_msg)

    except Exception as e:
        logger.error(f"File processing error: {e}")
        await progress_msg.edit_text(f"❌ प्रोसेसिंग त्रुटि: {e}", parse_mode="Markdown")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not config.api_key:
        await update.message.reply_text("❌ पहले `/setkey <key>` सेट करें।", parse_mode="Markdown")
        return

    progress_msg = await update.message.reply_text("⏳ **प्रगति:** फोटो डाउनलोड हो रहा है...", parse_mode="Markdown")
    photo = update.message.photo[-1]
    file_obj = await photo.get_file()
    file_path = await file_obj.download_to_drive()

    try:
        await progress_msg.edit_text("🖼️ **प्रगति:** फोटो का OCR हो रहा है...", parse_mode="Markdown")
        text_content = ocr_image(file_path)
        await process_text_and_generate_quiz(update, text_content, progress_msg)
    except Exception as e:
        logger.error(f"Photo processing error: {e}")
        await progress_msg.edit_text(f"❌ प्रोसेसिंग त्रुटि: {e}", parse_mode="Markdown")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Bot error: {context.error}")
    await update.message.reply_text(f"❌ कुछ गलत हो गया: {context.error}", parse_mode="Markdown")

def main():
    app = Application.builder().token("7779216065:AAEeAkWfwP0-HyTi2CQeJ6iULvGkW-EqamQ").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("setkey", set_key))
    app.add_handler(CommandHandler("listmodels", list_models))
    app.add_handler(CommandHandler("setmode", set_mode))
    app.add_handler(CommandHandler("setprompt", set_prompt))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(button_handler, pattern="^model_"))
    app.add_error_handler(error)
    app.run_polling()

if __name__ == "__main__":
    main()