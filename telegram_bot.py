# telegram_bot.py
import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode
import redis.asyncio as redis
import json
from datetime import datetime, timedelta
import hashlib
import uuid
from typing import Dict, Optional
import asyncpg
import aioredis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
import httpx

# ================ تنظیمات پیشرفته ================
BOT_TOKEN = "8052349235:AAFSaJmYpl359BKrJTWC8O-u-dI9r2olEOQ"
REDIS_URL = "redis://localhost:6379"
POSTGRES_DSN = "postgresql://user:pass@localhost/dbname"
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
            cls._instance.redis = None
        return cls._instance
    
    async def init(self):
        """ایجاد اتصال به دیتابیس"""
        self.pool = await asyncpg.create_pool(
            POSTGRES_DSN,
            min_size=10,
            max_size=100,
            command_timeout=60,
            max_queries=50000,
            max_inactive_connection_lifetime=300
        )
        
        self.redis = await aioredis.from_url(
            REDIS_URL,
            max_connections=50,
            decode_responses=True
        )
        
        # ایجاد جداول
        await self.init_tables()
    
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
                    is_admin BOOLEAN DEFAULT FALSE,
                    INDEX idx_users_referral (referral_code),
                    INDEX idx_users_active (last_active)
                )
            ''')
            
            # جدول دانش هوش مصنوعی
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS ai_knowledge (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    category TEXT,
                    question_vector BYTEA,
                    times_used INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    created_by BIGINT,
                    feedback JSONB DEFAULT '{"positive": 0, "negative": 0}',
                    INDEX idx_ai_search (question),
                    INDEX idx_ai_category (category),
                    INDEX idx_ai_used (times_used DESC)
                )
            ''')
            
            # جدول آگهی‌های شغلی
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id SERIAL PRIMARY KEY,
                    employer_id BIGINT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    salary TEXT,
                    location TEXT,
                    job_type TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    expires_at TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    views INTEGER DEFAULT 0,
                    applicants JSONB DEFAULT '[]',
                    INDEX idx_jobs_status (status),
                    INDEX idx_jobs_category (category),
                    INDEX idx_jobs_employer (employer_id),
                    INDEX idx_jobs_created (created_at DESC)
                )
            ''')
            
            # جدول رزومه‌ها
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS resumes (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    full_name TEXT,
                    skills TEXT[],
                    experience TEXT,
                    education TEXT,
                    desired_job TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    status TEXT DEFAULT 'active',
                    views INTEGER DEFAULT 0,
                    INDEX idx_resumes_user (user_id),
                    INDEX idx_resumes_status (status)
                )
            ''')
            
            # جدول سوالات بی‌پاسخ
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS unanswered_questions (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    user_id BIGINT,
                    asked_at TIMESTAMP DEFAULT NOW(),
                    answered BOOLEAN DEFAULT FALSE,
                    INDEX idx_unanswered (answered, asked_at)
                )
            ''')
    
    async def close(self):
        """بستن اتصالات"""
        if self.pool:
            await self.pool.close()
        if self.redis:
            await self.redis.close()

db = DatabasePool()

# ================ هسته هوش مصنوعی پیشرفته ================
class AIBrain:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            analyzer='char_wb',
            token_pattern=r'(?u)\b\w+\b'
        )
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.model_cache = {}
        
    async def search_smart(self, question: str, threshold: float = 0.3) -> list:
        """جستجوی هوشمند با کش"""
        # بررسی کش
        cache_key = f"search:{hashlib.md5(question.encode()).hexdigest()}"
        cached = await db.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # جستجو در دیتابیس
        async with db.pool.acquire() as conn:
            # دریافت دانش‌های پراستفاده اخیر
            knowledge = await conn.fetch('''
                SELECT id, question, answer, category 
                FROM ai_knowledge 
                WHERE times_used > 0 OR created_at > NOW() - INTERVAL '7 days'
                ORDER BY times_used DESC
                LIMIT 5000
            ''')
        
        if not knowledge:
            return []
        
        # پردازش موازی
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor,
            self._parallel_search,
            question,
            [dict(k) for k in knowledge]
        )
        
        # ذخیره در کش
        await db.redis.setex(cache_key, 300, json.dumps(results[:10]))
        
        return results[:10]
    
    def _parallel_search(self, question: str, knowledge: list) -> list:
        """جستجوی موازی با چند الگوریتم"""
        results = []
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        for item in knowledge:
            score = 0
            methods = []
            
            # الگوریتم 1: تطابق دقیق
            if item['question'].lower() == question_lower:
                score = 1.0
                methods.append('exact')
            
            # الگوریتم 2: کلمات کلیدی
            item_words = set(item['question'].lower().split())
            common = question_words.intersection(item_words)
            if common:
                keyword_score = len(common) / max(len(item_words), 1)
                score = max(score, keyword_score * 0.8)
                methods.append('keyword')
            
            # الگوریتم 3: طول سوال
            if abs(len(question) - len(item['question'])) < 10:
                score = max(score, 0.3)
                methods.append('length')
            
            if score >= threshold:
                results.append({
                    'id': item['id'],
                    'answer': item['answer'],
                    'score': round(score, 3),
                    'category': item.get('category', 'عمومی'),
                    'method': '+'.join(methods)
                })
        
        # مرتب‌سازی نهایی
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # به‌روزرسانی آمار (غیرهمزمان)
        asyncio.create_task(self._update_usage_stats([r['id'] for r in results[:5]]))
        
        return results
    
    async def _update_usage_stats(self, knowledge_ids: list):
        """به‌روزرسانی آمار استفاده"""
        if not knowledge_ids:
            return
        
        async with db.pool.acquire() as conn:
            await conn.execute('''
                UPDATE ai_knowledge 
                SET times_used = times_used + 1,
                    last_used = NOW()
                WHERE id = ANY($1::int[])
            ''', knowledge_ids)
    
    async def add_knowledge(self, question: str, answer: str, category: str, user_id: int) -> dict:
        """اضافه کردن دانش جدید"""
        async with db.pool.acquire() as conn:
            # بررسی تکراری
            exists = await conn.fetchval('''
                SELECT id FROM ai_knowledge 
                WHERE question ILIKE $1 OR question <-> $2 < 0.3
                LIMIT 1
            ''', question, question)
            
            if exists:
                return {'success': False, 'message': 'سوال مشابه قبلاً ثبت شده'}
            
            # درج دانش جدید
            knowledge_id = await conn.fetchval('''
                INSERT INTO ai_knowledge (question, answer, category, created_by)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            ''', question, answer, category, user_id)
            
            return {'success': True, 'id': knowledge_id}
    
    async def bulk_add(self, text: str, user_id: int) -> dict:
        """اضافه کردن گروهی دانش"""
        lines = text.strip().split('\n')
        added = []
        errors = []
        
        for line in lines:
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q, a = parts
                    result = await self.add_knowledge(
                        q.strip(), a.strip(), 'bulk', user_id
                    )
                    if result['success']:
                        added.append(q.strip())
                    else:
                        errors.append(f"{q[:30]}...: {result['message']}")
        
        return {'added': added, 'errors': errors}

# ================ سیستم مدیریت کاربران ================
class UserManager:
    async def get_or_create_user(self, user_id: int, username: str = None, 
                                  first_name: str = None, referrer_id: int = None) -> dict:
        """دریافت یا ایجاد کاربر با قفل توزیع شده"""
        # استفاده از Redis Lock برای جلوگیری از race condition
        lock_key = f"user_lock:{user_id}"
        async with db.redis.lock(lock_key, timeout=10):
            async with db.pool.acquire() as conn:
                user = await conn.fetchrow(
                    'SELECT * FROM users WHERE user_id = $1', user_id
                )
                
                if not user:
                    # ایجاد کاربر جدید
                    referral_code = await self._generate_unique_code(conn)
                    
                    user = await conn.fetchrow('''
                        INSERT INTO users (user_id, username, first_name, referral_code, referred_by)
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING *
                    ''', user_id, username, first_name, referral_code, referrer_id)
                    
                    if referrer_id:
                        await self._process_referral(conn, referrer_id, user_id)
                
                return dict(user)
    
    async def _generate_unique_code(self, conn) -> str:
        """تولید کد یکتا"""
        while True:
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            exists = await conn.fetchval(
                'SELECT 1 FROM users WHERE referral_code = $1', code
            )
            if not exists:
                return code
    
    async def _process_referral(self, conn, referrer_id: int, new_user_id: int):
        """پردازش رفرال"""
        await conn.execute('''
            UPDATE users 
            SET referral_count = referral_count + 1,
                points = points + 10
            WHERE user_id = $1
        ''', referrer_id)

# ================ وب‌اپلیکیشن پیشرفته ================
web_app = FastAPI(title="Telegram Mini App", version="1.0.0")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WebAppData(BaseModel):
    initData: str
    query_id: str = None
    user: dict = None

# ================ ربات تلگرام ================
class TelegramBot:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        self.ai_brain = AIBrain()
        self.user_manager = UserManager()
        self.setup_handlers()
    
    def setup_handlers(self):
        """تنظیم هندلرها"""
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """هندلر استارت با دکمه شیشه‌ای"""
        user = update.effective_user
        args = context.args
        
        referrer_id = args[0] if args else None
        
        # ثبت کاربر
        await self.user_manager.get_or_create_user(
            user.id, user.username, user.first_name, referrer_id
        )
        
        # ایجاد دکمه شیشه‌ای
        web_app_url = f"https://your-domain.com/app?user={user.id}"
        
        keyboard = [[
            InlineKeyboardButton(
                "✨ پلی ✨", 
                web_app=WebAppInfo(url=web_app_url)
            )
        ]]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_text = f"""
🎉 **به ربات هوشمند خوش آمدید {user.first_name}!**

برای شروع روی دکمه **پلی** کلیک کنید.
        """
        
        await update.message.reply_text(
            welcome_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش کلیک‌ها"""
        query = update.callback_query
        await query.answer()
        
        data = json.loads(query.data)
        action = data.get('action')
        
        if action == 'delete_message':
            await query.message.delete()
    
    def run(self):
        """اجرای ربات"""
        self.app.run_polling()

# ================ APIهای وب‌اپلیکیشن ================
@web_app.get("/api/user/{user_id}")
async def get_user(user_id: int):
    """دریافت اطلاعات کاربر"""
    async with db.pool.acquire() as conn:
        user = await conn.fetchrow('SELECT * FROM users WHERE user_id = $1', user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # آمار کاربر
        jobs_count = await conn.fetchval(
            'SELECT COUNT(*) FROM jobs WHERE employer_id = $1 AND status = $2',
            user_id, 'active'
        )
        
        resumes_count = await conn.fetchval(
            'SELECT COUNT(*) FROM resumes WHERE user_id = $1 AND status = $2',
            user_id, 'active'
        )
        
        result = dict(user)
        result['jobs_count'] = jobs_count
        result['resumes_count'] = resumes_count
        result['referral_link'] = f"https://t.me/signaliiii_bot?start={user['referral_code']}"
        
        return result

@web_app.post("/api/chat")
async def chat_with_ai(request: dict):
    """چت با هوش مصنوعی"""
    question = request.get('message')
    user_id = request.get('user_id')
    
    if not question:
        raise HTTPException(status_code=400, detail="Message is required")
    
    # جستجوی هوشمند
    results = await ai_brain.search_smart(question)
    
    if results:
        return {
            'answer': results[0]['answer'],
            'confidence': results[0]['score'],
            'found': True
        }
    
    # ثبت سوال بی‌پاسخ
    async with db.pool.acquire() as conn:
        await conn.execute('''
            INSERT INTO unanswered_questions (question, user_id)
            VALUES ($1, $2)
        ''', question, user_id)
    
    return {
        'answer': None,
        'found': False
    }

@web_app.post("/api/jobs")
async def create_job(job: dict):
    """ایجاد آگهی شغلی"""
    async with db.pool.acquire() as conn:
        job_id = await conn.fetchval('''
            INSERT INTO jobs (employer_id, title, description, category, salary, location, job_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
        ''', job['employer_id'], job['title'], job['description'],
            job.get('category'), job.get('salary'), job.get('location'), job.get('job_type'))
        
        return {'id': job_id, 'success': True}

@web_app.get("/api/jobs")
async def get_jobs(category: str = None, page: int = 1, limit: int = 20):
    """دریافت آگهی‌های شغلی"""
    offset = (page - 1) * limit
    
    async with db.pool.acquire() as conn:
        query = '''
            SELECT j.*, u.username, u.first_name 
            FROM jobs j
            LEFT JOIN users u ON j.employer_id = u.user_id
            WHERE j.status = 'active'
        '''
        params = []
        
        if category and category != 'all':
            query += ' AND j.category = $1'
            params.append(category)
        
        query += ' ORDER BY j.created_at DESC LIMIT $' + str(len(params) + 1) + ' OFFSET $' + str(len(params) + 2)
        params.extend([limit, offset])
        
        jobs = await conn.fetch(query, *params)
        
        # تبدیل به فرمت کارت
        result = []
        for job in jobs:
            result.append({
                'id': job['id'],
                'title': job['title'],
                'description': job['description'][:150] + '...' if len(job['description']) > 150 else job['description'],
                'category': job['category'],
                'salary': job['salary'],
                'location': job['location'],
                'created_at': job['created_at'].isoformat(),
                'employer': job['first_name'] or job['username'],
                'views': job['views']
            })
        
        return {'jobs': result, 'page': page, 'has_more': len(jobs) == limit}

@web_app.post("/api/resumes")
async def create_resume(resume: dict):
    """ایجاد رزومه"""
    async with db.pool.acquire() as conn:
        resume_id = await conn.fetchval('''
            INSERT INTO resumes (user_id, full_name, skills, experience, education, desired_job)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
        ''', resume['user_id'], resume['full_name'], resume['skills'],
            resume['experience'], resume['education'], resume['desired_job'])
        
        return {'id': resume_id, 'success': True}

@web_app.post("/api/admin/train")
async def admin_train(request: dict):
    """آموزش به ربات (فقط ادمین)"""
    user_id = request.get('admin_id')
    
    # بررسی ادمین بودن
    async with db.pool.acquire() as conn:
        is_admin = await conn.fetchval(
            'SELECT is_admin FROM users WHERE user_id = $1', user_id
        )
        
        if not is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        question = request.get('question')
        answer = request.get('answer')
        category = request.get('category', 'admin')
        
        result = await ai_brain.add_knowledge(question, answer, category, user_id)
        return result

@web_app.post("/api/admin/bulk-train")
async def admin_bulk_train(request: dict):
    """آموزش دسته‌جمعی با فایل"""
    user_id = request.get('admin_id')
    text = request.get('text')
    
    async with db.pool.acquire() as conn:
        is_admin = await conn.fetchval(
            'SELECT is_admin FROM users WHERE user_id = $1', user_id
        )
        
        if not is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        result = await ai_brain.bulk_add(text, user_id)
        return result

@web_app.get("/api/admin/stats")
async def admin_stats(admin_id: int):
    """آمار سیستم"""
    async with db.pool.acquire() as conn:
        is_admin = await conn.fetchval(
            'SELECT is_admin FROM users WHERE user_id = $1', admin_id
        )
        
        if not is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # آمار کلی
        total_users = await conn.fetchval('SELECT COUNT(*) FROM users')
        active_today = await conn.fetchval('''
            SELECT COUNT(*) FROM users 
            WHERE last_active > NOW() - INTERVAL '1 day'
        ''')
        
        total_knowledge = await conn.fetchval('SELECT COUNT(*) FROM ai_knowledge')
        total_jobs = await conn.fetchval('SELECT COUNT(*) FROM jobs WHERE status = $1', 'active')
        total_resumes = await conn.fetchval('SELECT COUNT(*) FROM resumes WHERE status = $1', 'active')
        
        unanswered = await conn.fetchval('''
            SELECT COUNT(*) FROM unanswered_questions 
            WHERE answered = false
        ''')
        
        # آمار استفاده
        top_used = await conn.fetch('''
            SELECT question, times_used 
            FROM ai_knowledge 
            ORDER BY times_used DESC 
            LIMIT 10
        ''')
        
        return {
            'users': {
                'total': total_users,
                'active_today': active_today
            },
            'ai': {
                'total_knowledge': total_knowledge,
                'unanswered': unanswered,
                'top_used': [dict(t) for t in top_used]
            },
            'jobs': {
                'total': total_jobs
            },
            'resumes': {
                'total': total_resumes
            }
        }

# ================ اجرای همزمان ربات و وب‌سرور ================
async def main():
    """تابع اصلی"""
    # اتصال به دیتابیس
    await db.init()
    
    # راه‌اندازی ربات
    bot = TelegramBot(BOT_TOKEN)
    
    # راه‌اندازی وب‌سرور
    config = uvicorn.Config(web_app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    # اجرای همزمان
    await asyncio.gather(
        server.serve(),
        asyncio.to_thread(bot.run)
    )

if __name__ == "__main__":
    asyncio.run(main())
