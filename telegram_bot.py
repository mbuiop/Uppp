# telegram_advanced_bot.py
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
import json
import os
from datetime import datetime
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

# ================ تنظیمات ================
BOT_TOKEN = "8052349235:AAFSAJmYp1359BKJrJTWC80-u-dI9r2o1EOQ"
ADMIN_IDS = [123456789]  # آیدی عددی خودتو اینجا بزن

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ================ مغز هوشمند پیشرفته (همون کد خودت) ================
class AdvancedHistoryBrain:
    def __init__(self, data_file='history_knowledge.json'):
        self.data_file = data_file
        self.knowledge_base = []
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.question_vectors = None
        self.unanswered_questions = []
        self.load_knowledge()
        self.update_vectors()
        
    def load_knowledge(self):
        """بارگذاری دانش"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            print(f"📚 {len(self.knowledge_base)} دانش بارگذاری شد")
        else:
            # نمونه اولیه
            sample_data = [
                {"id": 1, "question": "کوروش کبیر که بود", "answer": "کوروش بزرگ بنیانگذار شاهنشاهی هخامنشی بود", "category": "ایران باستان", "times_used": 0},
                {"id": 2, "question": "داریوش چه کرد", "answer": "داریوش بزرگ جاده شاهی را ساخت و امپراتوری را به ساتراپی‌ها تقسیم کرد", "category": "ایران باستان", "times_used": 0},
                {"id": 3, "question": "خشایارشا که بود", "answer": "خشایارشا پسر داریوش بزرگ بود که به یونان لشکر کشید", "category": "ایران باستان", "times_used": 0}
            ]
            self.knowledge_base = sample_data
            self.save_knowledge()
            
    def save_knowledge(self):
        """ذخیره دانش"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            
    def update_vectors(self):
        """به‌روزرسانی بردارهای سوالات"""
        if self.knowledge_base:
            questions = [item['question'] for item in self.knowledge_base]
            try:
                self.question_vectors = self.vectorizer.fit_transform(questions)
            except:
                self.question_vectors = None
                
    def preprocess_text(self, text):
        """پیش‌پردازش متن"""
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower()
        text = ' '.join([word for word in text.split() if len(word) > 1])
        return text
    
    def search_smart(self, query):
        """جستجوی هوشمند با TF-IDF (همون کد خودت)"""
        if not self.knowledge_base:
            return []
            
        query = self.preprocess_text(query)
        
        # جستجوی ساده کلمات کلیدی
        keyword_results = []
        query_words = set(query.split())
        
        for item in self.knowledge_base:
            question_words = set(item['question'].split())
            common_words = query_words.intersection(question_words)
            
            if common_words:
                score = len(common_words) / max(len(question_words), 1)
                if query == item['question']:
                    score = 1.0
                    
                keyword_results.append({
                    'id': item['id'],
                    'answer': item['answer'],
                    'score': score,
                    'category': item.get('category', 'عمومی')
                })
        
        # جستجوی برداری
        vector_results = []
        if self.question_vectors is not None and len(self.knowledge_base) > 0:
            try:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.question_vectors)[0]
                
                for i, score in enumerate(similarities):
                    if score > 0.1:
                        item = self.knowledge_base[i]
                        vector_results.append({
                            'id': item['id'],
                            'answer': item['answer'],
                            'score': float(score),
                            'category': item.get('category', 'عمومی')
                        })
            except:
                pass
        
        # ترکیب نتایج
        combined = {}
        for result in keyword_results + vector_results:
            rid = result['id']
            if rid not in combined or result['score'] > combined[rid]['score']:
                combined[rid] = result
                
        results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        
        # به‌روزرسانی آمار استفاده
        for result in results[:3]:
            for item in self.knowledge_base:
                if item['id'] == result['id']:
                    item['times_used'] = item.get('times_used', 0) + 1
                    item['last_used'] = datetime.now().isoformat()
                    break
                    
        self.save_knowledge()
        return results[:3]
    
    def add_knowledge(self, question, answer, category='عمومی'):
        """اضافه کردن دانش جدید"""
        # بررسی تکراری نبودن
        for item in self.knowledge_base:
            if item['question'].lower() == question.lower():
                return False, "این سوال قبلاً ثبت شده است"
                
        new_item = {
            'id': len(self.knowledge_base) + 1,
            'question': self.preprocess_text(question),
            'answer': answer,
            'category': category,
            'date_added': datetime.now().isoformat(),
            'times_used': 0,
            'last_used': None
        }
        
        self.knowledge_base.append(new_item)
        self.save_knowledge()
        self.update_vectors()
        return True, "دانش با موفقیت اضافه شد"
    
    def add_bulk_from_text(self, text, category='عمومی'):
        """اضافه کردن گروهی از متن"""
        lines = text.strip().split('\n')
        count = 0
        errors = []
        
        for line in lines:
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q, a = parts
                    success, msg = self.add_knowledge(q.strip(), a.strip(), category)
                    if success:
                        count += 1
                    else:
                        errors.append(f"خطا در {q}: {msg}")
                        
        return count, errors
    
    def record_unanswered(self, question):
        """ثبت سوالات بی‌پاسخ"""
        self.unanswered_questions.append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # ذخیره سوالات بی‌پاسخ
        with open('unanswered.json', 'w', encoding='utf-8') as f:
            json.dump(self.unanswered_questions[-100:], f, ensure_ascii=False, indent=2)
    
    def get_stats(self):
        """گرفتن آمار"""
        total = len(self.knowledge_base)
        if total == 0:
            return {}
            
        categories = Counter([item.get('category', 'عمومی') for item in self.knowledge_base])
        most_used = sorted(self.knowledge_base, key=lambda x: x.get('times_used', 0), reverse=True)[:5]
        never_used = len([item for item in self.knowledge_base if item.get('times_used', 0) == 0])
        
        return {
            'total': total,
            'categories': dict(categories),
            'most_used': most_used,
            'never_used_count': never_used,
            'unanswered_count': len(self.unanswered_questions)
        }

# ================ ساختن مغز ربات ================
brain = AdvancedHistoryBrain()

# ================ دستورات ربات ================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """دستور start"""
    user = update.effective_user
    
    # دکمه پلی برای وب‌اپ
    web_app_url = "https://your-domain.com"  # آدرس سایت خودتو بزن
    
    keyboard = [[
        InlineKeyboardButton(
            "✨ پلی ✨", 
            web_app=WebAppInfo(url=web_app_url)
        )
    ]]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_text = f"""
🎉 **سلام {user.first_name}!**

🧠 من ربات تاریخ‌دان هوشمند هستم!
هر سوال تاریخی داری بپرس.

📊 آمار فعلی: {brain.get_stats().get('total', 0)} دانش تاریخی

برای شروع سوالتو بپرس یا روی دکمه پلی بزن!
    """
    
    await update.message.reply_text(
        welcome_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر پیام‌ها (همون الگوریتم خودت)"""
    question = update.message.text
    
    # نشون بده داره تایپ میکنه
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # جستجوی هوشمند
    results = brain.search_smart(question)
    
    if results:
        best = results[0]
        confidence = int(best['score'] * 100)
        
        response = f"""
🔍 **پاسخ:**

{best['answer']}

---
📊 دقت: {confidence}%
📂 دسته: {best.get('category', 'عمومی')}
        """
    else:
        # ثبت سوال بی‌پاسخ
        brain.record_unanswered(question)
        
        response = """
❌ متأسفم! نتونستم جوابی پیدا کنم.

📝 این سوال برای مدیر ارسال شد.
از طریق پنل مدیریت می‌توانی به من یاد بدهی.
        """
    
    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """نمایش آمار"""
    stats = brain.get_stats()
    
    text = f"""
📊 **آمار ربات**

📚 کل دانش: {stats.get('total', 0)}
❓ سوالات بی‌پاسخ: {stats.get('unanswered_count', 0)}
📭 استفاده نشده: {stats.get('never_used_count', 0)}

📂 **دسته‌بندی‌ها:**
    """
    
    for cat, count in stats.get('categories', {}).items():
        text += f"\n• {cat}: {count} مورد"
    
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def teach_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """آموزش به ربات (فقط ادمین)"""
    user_id = update.effective_user.id
    
    # چک کردن ادمین
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("⛔ این دستور فقط برای ادمین است!")
        return
    
    # گرفتن متن آموزش
    text = update.message.text.replace('/teach', '').strip()
    
    if '|' in text:
        q, a = text.split('|', 1)
        success, msg = brain.add_knowledge(q.strip(), a.strip())
        if success:
            await update.message.reply_text(f"✅ {msg}")
        else:
            await update.message.reply_text(f"❌ {msg}")
    else:
        await update.message.reply_text(
            "❗ فرمت صحیح:\n"
            "/teach سوال | جواب"
        )

async def bulk_teach(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """آموزش دسته‌جمعی (فقط ادمین)"""
    user_id = update.effective_user.id
    
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("⛔ این دستور فقط برای ادمین است!")
        return
    
    text = update.message.text.replace('/bulk', '').strip()
    
    if text:
        count, errors = brain.add_bulk_from_text(text)
        response = f"✅ {count} مورد اضافه شد"
        if errors:
            response += f"\n❌ خطاها:\n" + "\n".join(errors[:3])
        await update.message.reply_text(response)
    else:
        await update.message.reply_text(
            "❗ لطفاً متن را ارسال کنید.\n"
            "فرمت: سوال | جواب (هر خط یک مورد)"
        )

async def unanswered_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """نمایش سوالات بی‌پاسخ (فقط ادمین)"""
    user_id = update.effective_user.id
    
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("⛔ این دستور فقط برای ادمین است!")
        return
    
    unanswered = brain.unanswered_questions[-10:]  # ۱۰ تای آخر
    
    if not unanswered:
        await update.message.reply_text("✅ هیچ سوال بی‌پاسخی وجود ندارد!")
        return
    
    text = "❓ **سوالات بی‌پاسخ اخیر:**\n\n"
    for i, q in enumerate(unanswered, 1):
        text += f"{i}. {q['question']}\n"
    
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """راهنما"""
    user_id = update.effective_user.id
    
    help_text = """
📚 **راهنمای ربات**

🔹 **دستورات عمومی:**
/start - شروع مجدد
/help - راهنما
/stats - آمار ربات

🔹 **دستورات مخصوص ادمین:**
/teach سوال | جواب - آموزش تکی
/bulk - آموزش دسته‌جمعی
/unanswered - سوالات بی‌پاسخ
    """
    
    if user_id in ADMIN_IDS:
        help_text += "\n👑 شما ادمین هستید!"
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

# ================ اجرای ربات ================
def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     🤖 ربات تاریخ‌دان هوشمند - نسخه اصلی                  ║
    ║     📚 برگرفته از کد Ghh.py                                ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📚 دانش فعلی: {len(brain.knowledge_base)} مورد")
    print(f"🤖 ربات در حال اجراست...\n")
    
    # ساختن ربات
    app = Application.builder().token(BOT_TOKEN).build()
    
    # اضافه کردن هندلرها
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("teach", teach_command))
    app.add_handler(CommandHandler("bulk", bulk_teach))
    app.add_handler(CommandHandler("unanswered", unanswered_command))
    
    # هندلر پیام‌های معمولی
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # اجرا
    app.run_polling()

if __name__ == '__main__':
    main()
