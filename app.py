from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

# فایل‌های دیتابیس
SITES_FILE = 'data/sites.json'
SIGNALS_FILE = 'data/signals.json'
CURRENCIES_FILE = 'data/currencies.json'
INDICATORS_FILE = 'data/indicators.json'

# ایجاد دایرکتوری داده اگر وجود نداشته باشد
os.makedirs('data', exist_ok=True)

# مقداردهی اولیه فایل‌های JSON اگر وجود نداشته باشند
def init_files():
    if not os.path.exists(SITES_FILE):
        with open(SITES_FILE, 'w') as f:
            json.dump([], f)
    
    if not os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, 'w') as f:
            json.dump([], f)
    
    if not os.path.exists(CURRENCIES_FILE):
        with open(CURRENCIES_FILE, 'w') as f:
            json.dump([], f)
    
    if not os.path.exists(INDICATORS_FILE):
        with open(INDICATORS_FILE, 'w') as f:
            json.dump([], f)

init_files()

# روت‌های سرور
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_site', methods=['POST'])
def submit_site():
    data = request.json
    with open(SITES_FILE, 'r+') as f:
        sites = json.load(f)
        sites.append(data)
        f.seek(0)
        json.dump(sites, f)
    return jsonify({'status': 'success'})

@app.route('/get_sites', methods=['GET'])
def get_sites():
    with open(SITES_FILE, 'r') as f:
        sites = json.load(f)
    return jsonify(sites)

@app.route('/get_signals', methods=['GET'])
def get_signals():
    with open(SIGNALS_FILE, 'r') as f:
        signals = json.load(f)
    return jsonify(signals)

@app.route('/get_currencies', methods=['GET'])
def get_currencies():
    with open(CURRENCIES_FILE, 'r') as f:
        currencies = json.load(f)
    return jsonify(currencies)

@app.route('/get_indicators', methods=['GET'])
def get_indicators():
    with open(INDICATORS_FILE, 'r') as f:
        indicators = json.load(f)
    return jsonify(indicators)

@app.route('/mo')
def mo():
    return render_template('mo.html')

if __name__ == '__main__':
    app.run(debug=True)