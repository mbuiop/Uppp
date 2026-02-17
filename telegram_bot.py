# telegram_bot.py
import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode
import redis
import json
from datetime import datetime, timedelta
import hashlib
import uuid
from typing import Dict, Optional
import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import httpx
import random
import string

# ================ تنظیمات ================
BOT_TOKEN = "8052349235:AAFSAJmYp1359BKJrJTWC80-u-dI9r2o1EQ0"
REDIS_URL = "redis://:botpass123@localhost:6379/0"
POSTGRES_DSN = "postgresql://botuser:botpass123@localhost/botdb"
MAX_WORKERS = 100
CACHE_TTL = 3600

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ================ اتصال به دیتابیس ================
class DatabasePool:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pool = None
            cls._instance.redis_client = None
        return cls._instance

    async def init(self):
        """ایجاد اتصال به دیتابیس"""
        try:
            # اتصال به PostgreSQL
            self.pool = await asyncpg.create_pool(
                POSTGRES_DSN,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # اتصال به Redis
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                password='botpass123',
                decode_responses=True,
                db=0
            )
            
            # ایجاد جداول
            await self.init_tables()
            logger.info("✅ اتصال به دیتابیس برقرار شد")
            return True
        except Exception as e:
            logger.error(f"❌ خطا در اتصال به دیتابیس: {e}")
            return False
    
    async def init_tables(self):
        """ایجاد جداول دیتابیس"""
        async with self.pool.acquire() as conn:
            # جدول کاربران
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    joined_at TIMESTAMP DEFAULT NOW(),
                    last_active TIMESTAMP DEFAULT NOW(),
                    referral_code TEXT UNIQUE,
                    referred_by BIGINT,
                    points INTEGER DEFAULT 0,
                    referral_count INTEGER DEFAULT 0,
                    settings JSONB DEFAULT '{}',
                    is_admin BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # جدول دانش هوش مصنوعی
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS ai_knowledge (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    category TEXT,
                    times_used INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    created_by BIGINT,
                    feedback JSONB DEFAULT '{"positive": 0, "negative": 0}'
                )
            ''')

db = DatabasePool()

# ================ هسته هوش مصنوعی ================
class AIBrain:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.knowledge = []
        
    async def search(self, question):
        """جستجوی ساده در دانش"""
        question = question.lower()
        for item in self.knowledge:
            if item['question'].lower() in question or question in item['question'].lower():
                return item['answer']
        return None

ai_brain = AIBrain()

# ================ ربات تلگرام ================
class TelegramBot:
    def __init__(self, token):
        self.token = token
        self.app = Application.builder().token(token).build()
        self.setup_handlers()
    
    def setup_handlers(self):
        """تنظیم هندلرها"""
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CommandHandler("admin", self.admin_panel))
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دستور start"""
        user = update.effective_user
        
        # دکمه پلی
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

به ربات هوشمند خوش اومدی!

✨ **امکانات:**
🤖 چت با هوش مصنوعی
💼 ثبت آگهی شغلی
📝 ثبت رزومه
🎁 سیستم دعوت

برای شروع روی دکمه **پلی** کلیک کن!
        """
        
        await update.message.reply_text(
            welcome_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دستور help"""
        help_text = """
📚 **راهنمای ربات**

🔹 **دستورات:**
/start - شروع مجدد
/help - راهنما
/admin - پنل مدیریت (فقط ادمین)

🔹 **امکانات:**
• هوش مصنوعی پیشرفته
• ثبت آگهی شغلی
• ساخت رزومه
• دعوت از دوستان
        """
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def admin_panel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پنل مدیریت"""
        user_id = update.effective_user.id
        
        # چک کردن ادمین
        if user_id != 123456789:  # آیدی خودتو اینجا بزن
            await update.message.reply_text("⛔ دسترسی غیرمجاز!")
            return
        
        text = """
⚙️ **پنل مدیریت**

📊 آمار سیستم
🎓 آموزش ربات
👥 مدیریت کاربران
📁 آپلود فایل
        """
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    
    def run(self):
        """اجرای ربات"""
        print("""
        ╔════════════════════════════════════════════════════════════╗
        ║     🤖 ربات هوشمند - در حال راه‌اندازی...                 ║
        ╚════════════════════════════════════════════════════════════╝
        """)
        
        print(f"✅ ربات با موفقیت راه‌اندازی شد!")
        print(f"📍 آدرس: https://t.me/{self.token.split(':')[0]}")
        print("⏳ منتظر پیام‌ها...\n")
        
        self.app.run_polling()

# ================ وب‌اپلیکیشن ================
web_app = FastAPI(title="ربات هوشمند")

@web_app.get("/")
async def root():
    return {"message": "ربات هوشمند فعال است"}

@web_app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

# ================ اجرای اصلی ================
async def main():
    """تابع اصلی"""
    # اتصال به دیتابیس
    db_connected = await db.init()
    if not db_connected:
        logger.warning("⚠️ ربات بدون دیتابیس اجرا می‌شود")
    
    # راه‌اندازی ربات
    bot = TelegramBot(BOT_TOKEN)
    
    # اجرا
    await asyncio.gather(
        asyncio.to_thread(bot.run)
    )

if __name__ == "__main__":
    asyncio.run(main())
