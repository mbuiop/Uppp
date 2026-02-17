# advanced_telegram_bot.py
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, 
    CallbackQueryHandler, ConversationHandler, ContextTypes
)
from telegram.constants import ParseMode
import json
import os
import hashlib
from datetime import datetime, timedelta
import random
import string
import asyncio
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import jdatetime
from typing import Dict, List, Optional
import aiofiles

# ================ تنظیمات ================
BOT_TOKEN = "8052349235:AAFSaJmYpl359BKrJTWC8O-u-dI9r2olEOQ"  # توکن ربات خود را اینجا قرار دهید
ADMIN_IDS = [327855654]  # آیدی ادمین‌ها
MAX_MESSAGE_LENGTH = 4096

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ================ وضعیت‌های مکالمه ================
(
    MAIN_MENU, ADD_JOB, ADD_RESUME, SEARCH_JOBS,
    TRAIN_BOT, BULK_TRAIN, AWAIT_MESSAGE, AWAIT_RESPONSE,
    JOB_DETAILS, RESUME_DETAILS, CONFIRM_DELETE
) = range(11)

# ================ مغز هوش مصنوعی پیشرفته ================
class AdvancedAIBrain:
    def __init__(self, data_file='data/ai_knowledge.json'):
        self.data_file = data_file
        self.knowledge_base = []
        self.user_conversations = defaultdict(list)
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        self.question_vectors = None
        self.unanswered_questions = []
        self.load_knowledge()
        self.update_vectors()
        
    def load_knowledge(self):
        """بارگذاری دانش"""
        os.makedirs('data', exist_ok=True)
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = []
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
        """پیش‌پردازش پیشرفته متن"""
        # حذف کاراکترهای خاص
        text = re.sub(r'[^\w\sآ-ی]', ' ', text)
        # حذف فاصله‌های اضافی
        text = ' '.join(text.split())
        return text.lower()
    
    def calculate_similarity(self, text1, text2):
        """محاسبه شباهت دو متن"""
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        
        # شباهت کلمات
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0
        
        # شباهت ترتیبی
        if text1 == text2:
            return 1.0
            
        return jaccard
    
    def search_smart(self, query, threshold=0.3):
        """جستجوی هوشمند با چندین الگوریتم"""
        if not self.knowledge_base:
            return []
            
        query = self.preprocess_text(query)
        results = []
        
        # الگوریتم 1: تطابق دقیق
        for item in self.knowledge_base:
            if item['question'] == query:
                return [{
                    'id': item['id'],
                    'answer': item['answer'],
                    'score': 1.0,
                    'category': item.get('category', 'عمومی'),
                    'method': 'exact_match'
                }]
        
        # الگوریتم 2: شباهت کلمات کلیدی
        query_words = set(query.split())
        for item in self.knowledge_base:
            item_words = set(item['question'].split())
            common_words = query_words.intersection(item_words)
            
            if common_words:
                score = len(common_words) / max(len(item_words), 1)
                score *= 1.2  # وزن بیشتر برای کلمات کلیدی
                
                if score > threshold:
                    results.append({
                        'id': item['id'],
                        'answer': item['answer'],
                        'score': score,
                        'category': item.get('category', 'عمومی'),
                        'method': 'keyword'
                    })
        
        # الگوریتم 3: جستجوی برداری (TF-IDF)
        if self.question_vectors is not None:
            try:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.question_vectors)[0]
                
                for i, score in enumerate(similarities):
                    if score > threshold:
                        item = self.knowledge_base[i]
                        # بررسی تکراری نبودن
                        exists = any(r['id'] == item['id'] for r in results)
                        if not exists:
                            results.append({
                                'id': item['id'],
                                'answer': item['answer'],
                                'score': float(score),
                                'category': item.get('category', 'عمومی'),
                                'method': 'vector'
                            })
                        else:
                            # به‌روزرسانی امتیاز اگر بهتر است
                            for r in results:
                                if r['id'] == item['id'] and score > r['score']:
                                    r['score'] = float(score)
                                    r['method'] = 'vector_improved'
            except:
                pass
        
        # مرتب‌سازی بر اساس امتیاز
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # به‌روزرسانی آمار استفاده
        for result in results[:5]:
            for item in self.knowledge_base:
                if item['id'] == result['id']:
                    item['times_used'] = item.get('times_used', 0) + 1
                    item['last_used'] = datetime.now().isoformat()
                    break
        
        self.save_knowledge()
        return results[:5]
    
    def add_knowledge(self, question, answer, category='عمومی', added_by=None):
        """اضافه کردن دانش جدید"""
        # بررسی تکراری نبودن
        for item in self.knowledge_base:
            if self.calculate_similarity(item['question'], question) > 0.8:
                return False, "این سوال مشابه قبلاً ثبت شده است"
        
        new_id = len(self.knowledge_base) + 1
        new_item = {
            'id': new_id,
            'question': self.preprocess_text(question),
            'original_question': question,
            'answer': answer,
            'category': category,
            'added_by': added_by,
            'date_added': datetime.now().isoformat(),
            'times_used': 0,
            'last_used': None,
            'feedback': {'positive': 0, 'negative': 0}
        }
        
        self.knowledge_base.append(new_item)
        self.save_knowledge()
        self.update_vectors()
        return True, f"دانش با ID {new_id} اضافه شد"
    
    def add_bulk_from_text(self, text, category='عمومی', added_by=None):
        """اضافه کردن گروهی از متن"""
        lines = text.strip().split('\n')
        added = []
        errors = []
        
        for line in lines:
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q, a = parts
                    success, msg = self.add_knowledge(q.strip(), a.strip(), category, added_by)
                    if success:
                        added.append(q.strip())
                    else:
                        errors.append(f"{q}: {msg}")
        
        return added, errors
    
    def record_unanswered(self, question, user_id):
        """ثبت سوالات بی‌پاسخ"""
        self.unanswered_questions.append({
            'question': question,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # ذخیره سوالات بی‌پاسخ
        with open('data/unanswered.json', 'w', encoding='utf-8') as f:
            json.dump(self.unanswered_questions[-200:], f, ensure_ascii=False, indent=2)
    
    def get_stats(self):
        """گرفتن آمار"""
        if not self.knowledge_base:
            return {}
            
        categories = Counter([item.get('category', 'عمومی') for item in self.knowledge_base])
        most_used = sorted(self.knowledge_base, key=lambda x: x.get('times_used', 0), reverse=True)[:10]
        never_used = [item for item in self.knowledge_base if item.get('times_used', 0) == 0]
        
        return {
            'total': len(self.knowledge_base),
            'categories': dict(categories),
            'most_used': most_used,
            'never_used_count': len(never_used),
            'unanswered_count': len(self.unanswered_questions),
            'total_usage': sum(item.get('times_used', 0) for item in self.knowledge_base)
        }

# ================ سیستم مدیریت کاربران ================
class UserManager:
    def __init__(self):
        self.users_file = 'data/users.json'
        self.referrals_file = 'data/referrals.json'
        self.jobs_file = 'data/jobs.json'
        self.resumes_file = 'data/resumes.json'
        self.load_data()
        
    def load_data(self):
        """بارگذاری اطلاعات"""
        os.makedirs('data', exist_ok=True)
        
        # کاربران
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r', encoding='utf-8') as f:
                self.users = json.load(f)
        else:
            self.users = {}
            
        # رفرال‌ها
        if os.path.exists(self.referrals_file):
            with open(self.referrals_file, 'r', encoding='utf-8') as f:
                self.referrals = json.load(f)
        else:
            self.referrals = {}
            
        # شغل‌ها
        if os.path.exists(self.jobs_file):
            with open(self.jobs_file, 'r', encoding='utf-8') as f:
                self.jobs = json.load(f)
        else:
            self.jobs = []
            
        # رزومه‌ها
        if os.path.exists(self.resumes_file):
            with open(self.resumes_file, 'r', encoding='utf-8') as f:
                self.resumes = json.load(f)
        else:
            self.resumes = []
    
    def save_users(self):
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False, indent=2)
    
    def save_referrals(self):
        with open(self.referrals_file, 'w', encoding='utf-8') as f:
            json.dump(self.referrals, f, ensure_ascii=False, indent=2)
    
    def save_jobs(self):
        with open(self.jobs_file, 'w', encoding='utf-8') as f:
            json.dump(self.jobs, f, ensure_ascii=False, indent=2)
    
    def save_resumes(self):
        with open(self.resumes_file, 'w', encoding='utf-8') as f:
            json.dump(self.resumes, f, ensure_ascii=False, indent=2)
    
    def get_or_create_user(self, user_id, username=None, first_name=None, referrer_id=None):
        """دریافت یا ایجاد کاربر جدید"""
        user_id = str(user_id)
        
        if user_id not in self.users:
            # ایجاد کاربر جدید
            referral_code = self.generate_referral_code()
            
            self.users[user_id] = {
                'id': user_id,
                'username': username,
                'first_name': first_name,
                'joined_date': datetime.now().isoformat(),
                'last_active': datetime.now().isoformat(),
                'referral_code': referral_code,
                'referred_by': str(referrer_id) if referrer_id else None,
                'referral_count': 0,
                'points': 0,
                'jobs_posted': [],
                'resumes_posted': [],
                'settings': {
                    'language': 'fa',
                    'notifications': True
                },
                'stats': {
                    'messages_sent': 0,
                    'commands_used': 0,
                    'trainings_done': 0
                }
            }
            
            # ثبت رفرال
            if referrer_id:
                self.add_referral(referrer_id, user_id)
            
            self.save_users()
            logger.info(f"کاربر جدید: {user_id}")
        
        return self.users[user_id]
    
    def generate_referral_code(self, length=8):
        """تولید کد رفرال یکتا"""
        while True:
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
            # بررسی یکتا بودن
            if not any(u.get('referral_code') == code for u in self.users.values()):
                return code
    
    def add_referral(self, referrer_id, new_user_id):
        """ثبت رفرال جدید"""
        referrer_id = str(referrer_id)
        new_user_id = str(new_user_id)
        
        if referrer_id not in self.referrals:
            self.referrals[referrer_id] = []
        
        if new_user_id not in self.referrals[referrer_id]:
            self.referrals[referrer_id].append({
                'user_id': new_user_id,
                'date': datetime.now().isoformat(),
                'points_earned': 10
            })
            
            # افزایش امتیاز معرف
            if referrer_id in self.users:
                self.users[referrer_id]['referral_count'] += 1
                self.users[referrer_id]['points'] += 10
            
            self.save_referrals()
            self.save_users()
    
    def add_job(self, employer_id, title, description, category, salary=None, location=None):
        """اضافه کردن آگهی شغلی"""
        job = {
            'id': len(self.jobs) + 1,
            'employer_id': str(employer_id),
            'title': title,
            'description': description,
            'category': category,
            'salary': salary,
            'location': location,
            'date_posted': datetime.now().isoformat(),
            'status': 'active',
            'applicants': []
        }
        
        self.jobs.append(job)
        self.save_jobs()
        
        # به‌روزرسانی آمار کاربر
        if str(employer_id) in self.users:
            if 'jobs_posted' not in self.users[str(employer_id)]:
                self.users[str(employer_id)]['jobs_posted'] = []
            self.users[str(employer_id)]['jobs_posted'].append(job['id'])
            self.save_users()
        
        return job
    
    def add_resume(self, user_id, full_name, skills, experience, education, desired_job):
        """اضافه کردن رزومه"""
        resume = {
            'id': len(self.resumes) + 1,
            'user_id': str(user_id),
            'full_name': full_name,
            'skills': skills,
            'experience': experience,
            'education': education,
            'desired_job': desired_job,
            'date_posted': datetime.now().isoformat(),
            'status': 'active',
            'views': 0
        }
        
        self.resumes.append(resume)
        self.save_resumes()
        
        # به‌روزرسانی آمار کاربر
        if str(user_id) in self.users:
            if 'resumes_posted' not in self.users[str(user_id)]:
                self.users[str(user_id)]['resumes_posted'] = []
            self.users[str(user_id)]['resumes_posted'].append(resume['id'])
            self.save_users()
        
        return resume
    
    def get_user_stats(self, user_id):
        """گرفتن آمار کاربر"""
        user_id = str(user_id)
        if user_id not in self.users:
            return {}
        
        user = self.users[user_id]
        
        # آمار رفرال
        referrals = self.referrals.get(user_id, [])
        
        # آمار شغل‌ها
        user_jobs = [j for j in self.jobs if j.get('employer_id') == user_id]
        active_jobs = [j for j in user_jobs if j.get('status') == 'active']
        
        # آمار رزومه‌ها
        user_resumes = [r for r in self.resumes if r.get('user_id') == user_id]
        
        return {
            'points': user.get('points', 0),
            'referral_count': len(referrals),
            'referral_code': user.get('referral_code'),
            'jobs_count': len(user_jobs),
            'active_jobs': len(active_jobs),
            'resumes_count': len(user_resumes),
            'joined_date': user.get('joined_date')
        }
    
    def delete_job(self, job_id, user_id):
        """حذف آگهی شغلی (فقط توسط صاحب آگهی یا ادمین)"""
        job_id = int(job_id)
        user_id = str(user_id)
        
        for i, job in enumerate(self.jobs):
            if job['id'] == job_id:
                if job['employer_id'] == user_id or self.is_admin(user_id):
                    self.jobs[i]['status'] = 'deleted'
                    self.save_jobs()
                    return True, "آگهی با موفقیت حذف شد"
                else:
                    return False, "شما اجازه حذف این آگهی را ندارید"
        
        return False, "آگهی یافت نشد"
    
    def delete_user(self, admin_id, target_user_id):
        """حذف کاربر (فقط توسط ادمین)"""
        if not self.is_admin(admin_id):
            return False, "شما ادمین نیستید"
        
        target_user_id = str(target_user_id)
        if target_user_id in self.users:
            # غیرفعال کردن کاربر به جای حذف کامل
            self.users[target_user_id]['status'] = 'deleted'
            self.save_users()
            return True, f"کاربر {target_user_id} حذف شد"
        
        return False, "کاربر یافت نشد"
    
    def is_admin(self, user_id):
        """بررسی ادمین بودن"""
        return str(user_id) in [str(admin_id) for admin_id in ADMIN_IDS]

# ================ ربات اصلی ================
class AdvancedTelegramBot:
    def __init__(self, token):
        self.token = token
        self.app = Application.builder().token(token).build()
        self.ai_brain = AdvancedAIBrain()
        self.user_manager = UserManager()
        self.setup_handlers()
        
    def setup_handlers(self):
        """تنظیم تمام هندلرها"""
        
        # ========== دستورات پایه ==========
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("menu", self.main_menu))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("reset", self.reset_chat))
        
        # ========== سیستم رفرال ==========
        self.app.add_handler(CommandHandler("referral", self.referral_info))
        self.app.add_handler(CommandHandler("points", self.points_info))
        
        # ========== منوی اصلی ==========
        self.app.add_handler(MessageHandler(filters.Regex('^(🏠 منوی اصلی)$'), self.main_menu))
        self.app.add_handler(MessageHandler(filters.Regex('^(🤖 چت با هوش مصنوعی)$'), self.ai_chat_mode))
        self.app.add_handler(MessageHandler(filters.Regex('^(💼 ثبت شغل)$'), self.add_job_start))
        self.app.add_handler(MessageHandler(filters.Regex('^(📝 ثبت رزومه)$'), self.add_resume_start))
        self.app.add_handler(MessageHandler(filters.Regex('^(🔍 جستجوی کار)$'), self.search_jobs_start))
        self.app.add_handler(MessageHandler(filters.Regex('^(📋 کارهای من)$'), self.my_jobs))
        self.app.add_handler(MessageHandler(filters.Regex('^(👤 پروفایل من)$'), self.my_profile))
        
        # ========== دکمه‌های مدیریتی ==========
        self.app.add_handler(MessageHandler(filters.Regex('^(⚙️ پنل مدیریت)$'), self.admin_panel))
        self.app.add_handler(MessageHandler(filters.Regex('^(📊 آمار)$'), self.admin_stats))
        self.app.add_handler(MessageHandler(filters.Regex('^(🎓 آموزش ربات)$'), self.train_bot_start))
        self.app.add_handler(MessageHandler(filters.Regex('^(📁 آپلود دسته‌جمعی)$'), self.bulk_train_start))
        self.app.add_handler(MessageHandler(filters.Regex('^(❓ سوالات بی‌پاسخ)$'), self.unanswered_questions))
        self.app.add_handler(MessageHandler(filters.Regex('^(👥 مدیریت کاربران)$'), self.manage_users))
        
        # ========== هندلرهای مکالمه ==========
        conv_handler = ConversationHandler(
            entry_points=[
                MessageHandler(filters.Regex('^(💼 ثبت شغل)$'), self.add_job_start),
                MessageHandler(filters.Regex('^(📝 ثبت رزومه)$'), self.add_resume_start),
                MessageHandler(filters.Regex('^(🎓 آموزش ربات)$'), self.train_bot_start),
                MessageHandler(filters.Regex('^(📁 آپلود دسته‌جمعی)$'), self.bulk_train_start),
            ],
            states={
                ADD_JOB: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.add_job_title)
                ],
                ADD_RESUME: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.add_resume_name)
                ],
                TRAIN_BOT: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.train_bot_question)
                ],
                BULK_TRAIN: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.bulk_train_process)
                ],
            },
            fallbacks=[CommandHandler("cancel", self.cancel)],
        )
        self.app.add_handler(conv_handler)
        
        # ========== هندلر پیام‌های عادی ==========
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # ========== هندلر Callback Query ==========
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دستور start با پشتیبانی از رفرال"""
        user = update.effective_user
        args = context.args
        
        referrer_id = args[0] if args else None
        
        # ثبت یا دریافت کاربر
        db_user = self.user_manager.get_or_create_user(
            user.id, 
            user.username, 
            user.first_name,
            referrer_id
        )
        
        welcome_text = f"""
🎉 به ربات هوشمند خوش آمدید {user.first_name}!

🧠 این ربات با هوش مصنوعی پیشرفته می‌تواند:
• به سوالات شما پاسخ دهد
• آگهی شغلی ثبت کند
• رزومه ثبت کند
• کار پیدا کند
• و خیلی چیزهای دیگر!

📌 برای شروع از منوی اصلی استفاده کنید.
        """
        
        # ایجاد منوی اصلی
        reply_markup = self.get_main_menu(user.id)
        
        await update.message.reply_text(welcome_text, reply_markup=reply_markup)
    
    def get_main_menu(self, user_id):
        """ایجاد منوی اصلی بر اساس سطح دسترسی"""
        keyboard = [
            [KeyboardButton("🤖 چت با هوش مصنوعی")],
            [KeyboardButton("💼 ثبت شغل"), KeyboardButton("📝 ثبت رزومه")],
            [KeyboardButton("🔍 جستجوی کار"), KeyboardButton("📋 کارهای من")],
            [KeyboardButton("👤 پروفایل من"), KeyboardButton("🏠 منوی اصلی")]
        ]
        
        # دکمه‌های مدیریتی برای ادمین
        if self.user_manager.is_admin(user_id):
            keyboard.append([KeyboardButton("⚙️ پنل مدیریت")])
        
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    async def main_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش منوی اصلی"""
        user_id = update.effective_user.id
        reply_markup = self.get_main_menu(user_id)
        await update.message.reply_text("🏠 منوی اصلی:", reply_markup=reply_markup)
    
    async def ai_chat_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """فعال کردن حالت چت با هوش مصنوعی"""
        context.user_data['mode'] = 'ai_chat'
        await update.message.reply_text(
            "🧠 حالت چت با هوش مصنوعی فعال شد!\n"
            "هر سوالی دارید بپرسید. برای بازگشت به منو /menu را بزنید."
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """هندلر اصلی پیام‌ها"""
        user = update.effective_user
        message = update.message.text
        
        # به‌روزرسانی آخرین فعالیت
        if str(user.id) in self.user_manager.users:
            self.user_manager.users[str(user.id)]['last_active'] = datetime.now().isoformat()
            self.user_manager.save_users()
        
        # بررسی حالت فعلی
        mode = context.user_data.get('mode', 'normal')
        
        if mode == 'ai_chat':
            await self.handle_ai_chat(update, context)
        else:
            await update.message.reply_text(
                "لطفاً از منوی اصلی استفاده کنید.",
                reply_markup=self.get_main_menu(user.id)
            )
    
    async def handle_ai_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش چت با هوش مصنوعی"""
        question = update.message.text
        
        # نمایش تایپینگ
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # جستجو در دانش
        results = self.ai_brain.search_smart(question)
        
        if results:
            best_match = results[0]
            confidence = best_match['score'] * 100
            
            response = f"""
🔍 **پاسخ هوش مصنوعی:**

{best_match['answer']}

---
📊 دقت: {confidence:.1f}%
📂 دسته: {best_match.get('category', 'عمومی')}
📎 روش: {best_match.get('method', 'unknown')}
            """
        else:
            # ثبت سوال بی‌پاسخ
            self.ai_brain.record_unanswered(question, user.id)
            
            response = """
❌ متأسفم! هنوز جواب این سوال را یاد نگرفته‌ام.

📝 این سوال برای مدیر ارسال شد.
💡 می‌توانید از منوی آموزش، به من یاد بدهید!
            """
        
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    
    async def referral_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش اطلاعات رفرال"""
        user_id = update.effective_user.id
        db_user = self.user_manager.users.get(str(user_id), {})
        
        if not db_user:
            await update.message.reply_text("❌ کاربر یافت نشد!")
            return
        
        referral_code = db_user.get('referral_code')
        referral_link = f"https://t.me/{context.bot.username}?start={referral_code}"
        referral_count = db_user.get('referral_count', 0)
        points = db_user.get('points', 0)
        
        text = f"""
🎁 **سیستم دعوت از دوستان**

🔗 لینک دعوت شما:
`{referral_link}`

📊 آمار شما:
• تعداد دعوت‌ها: {referral_count}
• امتیاز کسب شده: {points}

💡 با هر دعوت ۱۰ امتیاز می‌گیرید!
✨ امتیازها برای خدمات ویژه استفاده می‌شود.
        """
        
        # دکمه اشتراک‌گذاری
        keyboard = [[
            InlineKeyboardButton("📤 اشتراک‌گذاری لینک", url=f"https://t.me/share/url?url={referral_link}")
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    async def points_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش اطلاعات امتیازات"""
        user_id = update.effective_user.id
        db_user = self.user_manager.users.get(str(user_id), {})
        
        if not db_user:
            await update.message.reply_text("❌ کاربر یافت نشد!")
            return
        
        points = db_user.get('points', 0)
        referral_count = db_user.get('referral_count', 0)
        
        text = f"""
💰 **کیف پول امتیاز**

💎 امتیاز فعلی: {points}
🎯 تعداد دعوت‌ها: {referral_count}

**روش‌های کسب امتیاز:**
• هر دعوت: ۱۰ امتیاز
• ثبت آگهی: ۵ امتیاز
• آموزش به ربات: ۳ امتیاز
• استفاده روزانه: ۱ امتیاز

**مصرف امتیاز:**
• نمایش ویژه آگهی: ۲۰ امتیاز
• پشتیبانی ویژه: ۳۰ امتیاز
• تبلیغ رزومه: ۱۵ امتیاز
        """
        
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    
    async def reset_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پاک کردن تاریخچه چت"""
        # پاک کردن پیام‌های قدیمی (حداکثر 100 پیام آخر)
        chat_id = update.effective_chat.id
        message_id = update.message.message_id
        
        try:
            # تلاش برای پاک کردن پیام‌ها (محدودیت تلگرام)
            for i in range(message_id - 50, message_id):
                try:
                    await context.bot.delete_message(chat_id, i)
                except:
                    pass
            
            await update.message.reply_text("✅ تاریخچه چت پاک شد!")
        except Exception as e:
            await update.message.reply_text("⚠️ برخی پیام‌ها پاک شدند، بقیه را خودتان پاک کنید.")
    
    # ========== سیستم ثبت شغل ==========
    async def add_job_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شروع ثبت آگهی شغلی"""
        context.user_data['job_data'] = {}
        await update.message.reply_text(
            "📝 **ثبت آگهی شغلی جدید**\n\n"
            "لطفاً عنوان شغل را وارد کنید:",
            parse_mode=ParseMode.MARKDOWN
        )
        return ADD_JOB
    
    async def add_job_title(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت عنوان شغل"""
        context.user_data['job_data']['title'] = update.message.text
        await update.message.reply_text(
            "📝 توضیحات کامل شغل را وارد کنید:"
        )
        context.user_data['job_step'] = 'description'
        return ADD_JOB
    
    async def add_job_description(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت توضیحات شغل"""
        context.user_data['job_data']['description'] = update.message.text
        await update.message.reply_text(
            "💰 حقوق پیشنهادی را وارد کنید (یا خالی بگذارید):"
        )
        context.user_data['job_step'] = 'salary'
        return ADD_JOB
    
    async def add_job_salary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت حقوق"""
        context.user_data['job_data']['salary'] = update.message.text if update.message.text != '-' else None
        await update.message.reply_text(
            "📍 محل کار را وارد کنید (یا خالی بگذارید):"
        )
        context.user_data['job_step'] = 'location'
        return ADD_JOB
    
    async def add_job_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت محل کار"""
        context.user_data['job_data']['location'] = update.message.text if update.message.text != '-' else None
        
        # دسته‌بندی شغل
        categories = [
            "فناوری اطلاعات",
            "فروش و بازاریابی",
            "خدمات مشتریان",
            "حسابداری و مالی",
            "آموزش",
            "پذیرایی و رستوران",
            "ساختمان",
            "تولید و صنعت",
            "بهداشت و درمان",
            "سایر"
        ]
        
        keyboard = [[InlineKeyboardButton(cat, callback_data=f"job_cat_{cat}")] for cat in categories]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "📂 دسته‌بندی شغل را انتخاب کنید:",
            reply_markup=reply_markup
        )
        
        return ConversationHandler.END
    
    # ========== ثبت رزومه ==========
    async def add_resume_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شروع ثبت رزومه"""
        context.user_data['resume_data'] = {}
        await update.message.reply_text(
            "📝 **ثبت رزومه جدید**\n\n"
            "لطفاً نام و نام خانوادگی خود را وارد کنید:",
            parse_mode=ParseMode.MARKDOWN
        )
        return ADD_RESUME
    
    async def add_resume_name(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت نام"""
        context.user_data['resume_data']['full_name'] = update.message.text
        await update.message.reply_text(
            "🔧 مهارت‌های خود را وارد کنید (با ویرگول جدا کنید):"
        )
        context.user_data['resume_step'] = 'skills'
        return ADD_RESUME
    
    async def add_resume_skills(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت مهارت‌ها"""
        context.user_data['resume_data']['skills'] = update.message.text
        await update.message.reply_text(
            "💼 سابقه کار خود را وارد کنید:"
        )
        context.user_data['resume_step'] = 'experience'
        return ADD_RESUME
    
    async def add_resume_experience(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت سابقه کار"""
        context.user_data['resume_data']['experience'] = update.message.text
        await update.message.reply_text(
            "🎓 تحصیلات خود را وارد کنید:"
        )
        context.user_data['resume_step'] = 'education'
        return ADD_RESUME
    
    async def add_resume_education(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت تحصیلات"""
        context.user_data['resume_data']['education'] = update.message.text
        await update.message.reply_text(
            "🎯 شغل مورد نظر خود را وارد کنید:"
        )
        context.user_data['resume_step'] = 'desired_job'
        return ADD_RESUME
    
    async def add_resume_desired_job(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت شغل مورد نظر و ذخیره نهایی"""
        context.user_data['resume_data']['desired_job'] = update.message.text
        
        # ذخیره رزومه
        resume = self.user_manager.add_resume(
            user_id=update.effective_user.id,
            **context.user_data['resume_data']
        )
        
        await update.message.reply_text(
            f"✅ رزومه شما با موفقیت ثبت شد!\n"
            f"🆔 کد رزومه: {resume['id']}",
            reply_markup=self.get_main_menu(update.effective_user.id)
        )
        
        return ConversationHandler.END
    
    # ========== آموزش به ربات ==========
    async def train_bot_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شروع آموزش به ربات"""
        if not self.user_manager.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ این بخش فقط برای ادمین‌ها قابل دسترسی است!")
            return ConversationHandler.END
        
        await update.message.reply_text(
            "🧠 **آموزش به ربات**\n\n"
            "لطفاً **سوال** را وارد کنید:",
            parse_mode=ParseMode.MARKDOWN
        )
        return TRAIN_BOT
    
    async def train_bot_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت سوال"""
        context.user_data['train_question'] = update.message.text
        await update.message.reply_text(
            "📝 حالا **جواب** را وارد کنید:",
            parse_mode=ParseMode.MARKDOWN
        )
        context.user_data['train_step'] = 'answer'
        return TRAIN_BOT
    
    async def train_bot_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت جواب و ذخیره"""
        question = context.user_data['train_question']
        answer = update.message.text
        
        # اضافه کردن به دانش
        success, msg = self.ai_brain.add_knowledge(
            question, 
            answer, 
            added_by=update.effective_user.id
        )
        
        if success:
            await update.message.reply_text(
                f"✅ **آموزش با موفقیت انجام شد!**\n{msg}",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(f"❌ {msg}")
        
        return ConversationHandler.END
    
    # ========== آموزش دسته‌جمعی ==========
    async def bulk_train_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شروع آموزش دسته‌جمعی"""
        if not self.user_manager.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ این بخش فقط برای ادمین‌ها قابل دسترسی است!")
            return ConversationHandler.END
        
        await update.message.reply_text(
            "📁 **آموزش دسته‌جمعی**\n\n"
            "فرمت مورد قبول:\n"
            "`سوال ۱ | جواب ۱`\n"
            "`سوال ۲ | جواب ۲`\n"
            "`سوال ۳ | جواب ۳`\n\n"
            "متن خود را ارسال کنید:",
            parse_mode=ParseMode.MARKDOWN
        )
        return BULK_TRAIN
    
    async def bulk_train_process(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش آموزش دسته‌جمعی"""
        text = update.message.text
        
        added, errors = self.ai_brain.add_bulk_from_text(
            text, 
            added_by=update.effective_user.id
        )
        
        response = f"✅ {len(added)} مورد با موفقیت اضافه شد.\n"
        if errors:
            response += f"⚠️ {len(errors)} خطا:\n" + "\n".join(errors[:5])
        
        await update.message.reply_text(response)
        return ConversationHandler.END
    
    # ========== پنل مدیریت ==========
    async def admin_panel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پنل مدیریت"""
        if not self.user_manager.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ دسترسی غیرمجاز!")
            return
        
        keyboard = [
            [KeyboardButton("📊 آمار"), KeyboardButton("🎓 آموزش ربات")],
            [KeyboardButton("📁 آپلود دسته‌جمعی"), KeyboardButton("❓ سوالات بی‌پاسخ")],
            [KeyboardButton("👥 مدیریت کاربران"), KeyboardButton("🏠 منوی اصلی")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            "⚙️ **پنل مدیریت**\n\n"
            "یکی از گزینه‌ها را انتخاب کنید:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def admin_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش آمار"""
        if not self.user_manager.is_admin(update.effective_user.id):
            return
        
        # آمار هوش مصنوعی
        ai_stats = self.ai_brain.get_stats()
        
        # آمار کاربران
        total_users = len(self.user_manager.users)
        active_today = sum(
            1 for u in self.user_manager.users.values()
            if datetime.fromisoformat(u.get('last_active', '2000-01-01')) > datetime.now() - timedelta(days=1)
        )
        
        # آمار شغل‌ها
        total_jobs = len(self.user_manager.jobs)
        active_jobs = sum(1 for j in self.user_manager.jobs if j.get('status') == 'active')
        
        text = f"""
📊 **آمار کلی سیستم**

🧠 **هوش مصنوعی:**
• کل دانش: {ai_stats.get('total', 0)}
• دسته‌بندی‌ها: {len(ai_stats.get('categories', {}))}
• سوالات بی‌پاسخ: {ai_stats.get('unanswered_count', 0)}
• استفاده کل: {ai_stats.get('total_usage', 0)}

👥 **کاربران:**
• کل کاربران: {total_users}
• فعال امروز: {active_today}

💼 **شغل‌ها:**
• کل آگهی‌ها: {total_jobs}
• آگهی‌های فعال: {active_jobs}
        """
        
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    
    async def unanswered_questions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش سوالات بی‌پاسخ"""
        if not self.user_manager.is_admin(update.effective_user.id):
            return
        
        unanswered = self.ai_brain.unanswered_questions[-20:]  # ۲۰ تای آخر
        
        if not unanswered:
            await update.message.reply_text("✅ هیچ سوال بی‌پاسخی وجود ندارد!")
            return
        
        text = "❓ **سوالات بی‌پاسخ:**\n\n"
        for i, q in enumerate(unanswered, 1):
            text += f"{i}. {q['question']}\n"
        
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    
    async def manage_users(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت کاربران"""
        if not self.user_manager.is_admin(update.effective_user.id):
            return
        
        # نمایش ۱۰ کاربر برتر
        top_users = sorted(
            self.user_manager.users.values(),
            key=lambda x: x.get('points', 0),
            reverse=True
        )[:10]
        
        text = "👥 **کاربران برتر:**\n\n"
        for i, user in enumerate(top_users, 1):
            name = user.get('first_name', 'بدون نام')
            points = user.get('points', 0)
            referrals = user.get('referral_count', 0)
            text += f"{i}. {name} | امتیاز: {points} | دعوت: {referrals}\n"
        
        text += "\n🔍 برای جستجوی کاربر آیدی او را وارد کنید."
        
        # ایجاد دکمه برای کاربران بعدی
        keyboard = [[InlineKeyboardButton("📋 همه کاربران", callback_data="list_all_users")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    # ========== هندلر Callback ==========
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش Callback Query"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data.startswith("job_cat_"):
            category = data.replace("job_cat_", "")
            # ذخیره دسته‌بندی و نهایی‌سازی ثبت شغل
            job_data = context.user_data.get('job_data', {})
            job_data['category'] = category
            
            # ذخیره در دیتابیس
            job = self.user_manager.add_job(
                employer_id=update.effective_user.id,
                **job_data
            )
            
            await query.edit_message_text(
                f"✅ آگهی شغلی با موفقیت ثبت شد!\n"
                f"🆔 کد آگهی: {job['id']}\n"
                f"📌 عنوان: {job['title']}"
            )
        
        elif data == "list_all_users":
            # نمایش همه کاربران (به صورت صفحه‌بندی)
            users_list = list(self.user_manager.users.values())
            text = "👥 **لیست همه کاربران:**\n\n"
            
            for i, user in enumerate(users_list[:20], 1):
                name = user.get('first_name', 'بدون نام')
                user_id = user.get('id', '?')
                status = user.get('status', 'active')
                text += f"{i}. {name} | آیدی: {user_id} | وضعیت: {status}\n"
            
            text += f"\n📊 کل کاربران: {len(users_list)}"
            
            await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN)
    
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """لغو عملیات"""
        await update.message.reply_text(
            "❌ عملیات لغو شد.",
            reply_markup=self.get_main_menu(update.effective_user.id)
        )
        return ConversationHandler.END
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """راهنما"""
        help_text = """
📚 **راهنمای ربات**

🤖 **چت با هوش مصنوعی**
سوالات خود را بپرسید تا پاسخ بگیرید.

💼 **ثبت شغل**
آگهی استخدام ثبت کنید.

📝 **ثبت رزومه**
برای کاریابی رزومه ثبت کنید.

🔍 **جستجوی کار**
آگهی‌های شغلی را ببینید.

🎁 **دعوت از دوستان**
با /referral لینک دعوت بگیرید.

⚙️ **دستورات ویژه:**
/reset - پاک کردن چت
/points - مشاهده امتیازات
/menu - منوی اصلی
        """
        
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def my_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پروفایل کاربر"""
        user_id = update.effective_user.id
        stats = self.user_manager.get_user_stats(user_id)
        
        if not stats:
            await update.message.reply_text("❌ کاربر یافت نشد!")
            return
        
        text = f"""
👤 **پروفایل شما**

🆔 آیدی: {user_id}
📅 تاریخ عضویت: {stats.get('joined_date', 'نامشخص')[:10]}

💰 امتیاز: {stats.get('points', 0)}
🎁 تعداد دعوت: {stats.get('referral_count', 0)}
🔗 کد دعوت: `{stats.get('referral_code', '')}`

💼 آگهی‌های شغلی: {stats.get('jobs_count', 0)}
📋 رزومه‌ها: {stats.get('resumes_count', 0)}
        """
        
        keyboard = [[
            InlineKeyboardButton("🎁 لینک دعوت", callback_data="show_referral"),
            InlineKeyboardButton("💰 امتیازات", callback_data="show_points")
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    async def my_jobs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش شغل‌های کاربر"""
        user_id = str(update.effective_user.id)
        
        user_jobs = [j for j in self.user_manager.jobs if j.get('employer_id') == user_id]
        
        if not user_jobs:
            await update.message.reply_text("📭 شما هنوز آگهی ثبت نکرده‌اید.")
            return
        
        text = "📋 **آگهی‌های شما:**\n\n"
        for job in user_jobs[-5:]:  # ۵ تای آخر
            status = "✅ فعال" if job.get('status') == 'active' else "❌ غیرفعال"
            text += f"🆔 {job['id']} | {job['title']} | {status}\n"
        
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    
    async def search_jobs_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شروع جستجوی کار"""
        # نمایش دسته‌بندی‌ها
        categories = set(j.get('category', 'سایر') for j in self.user_manager.jobs if j.get('status') == 'active')
        
        if not categories:
            await update.message.reply_text("📭 در حال حاضر آگهی فعالی وجود ندارد.")
            return
        
        keyboard = []
        row = []
        for i, cat in enumerate(categories, 1):
            row.append(InlineKeyboardButton(cat, callback_data=f"search_{cat}"))
            if i % 2 == 0:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)
        
        keyboard.append([InlineKeyboardButton("🔍 همه آگهی‌ها", callback_data="search_all")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🔍 **جستجوی کار**\n\n"
            "دسته‌بندی مورد نظر را انتخاب کنید:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def search_jobs_by_category(self, category):
        """جستجوی آگهی بر اساس دسته‌بندی"""
        jobs = [j for j in self.user_manager.jobs if j.get('status') == 'active']
        
        if category != "all":
            jobs = [j for j in jobs if j.get('category') == category]
        
        return jobs
    
    def run(self):
        """اجرای ربات"""
        logger.info("ربات با موفقیت راه‌اندازی شد!")
        self.app.run_polling()

# ================ اجرای برنامه ================
if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     🤖 ربات هوشمند تلگرام - نسخه نهایی                    ║
    ║     🔥 با قابلیت یادگیری و سیستم کاریابی پیشرفته         ║
    ╠════════════════════════════════════════════════════════════╣
    ║  📌 برای اجرا توکن ربات را در خط 22 قرار دهید            ║
    ║  📌 آیدی ادمین‌ها را در خط 23 وارد کنید                   ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # ایجاد ربات و اجرا
    bot = AdvancedTelegramBot(BOT_TOKEN)
    bot.run()
