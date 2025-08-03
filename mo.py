from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

# فایل‌های دیتابیس (همانند app.py)
SITES_FILE = 'data/sites.json'
SIGNALS_FILE = 'data/signals.json'
CURRENCIES_FILE = 'data/currencies.json'
INDICATORS_FILE = 'data/indicators.json'

@app.route('/')
def mo_dashboard():
    return render_template('mo.html')

@app.route('/approve_site', methods=['POST'])
def approve_site():
    site_id = request.json.get('id')
    with open(SITES_FILE, 'r+') as f:
        sites = json.load(f)
        for site in sites:
            if site['id'] == site_id:
                site['approved'] = True
        f.seek(0)
        json.dump(sites, f)
    return jsonify({'status': 'success'})

@app.route('/add_signal', methods=['POST'])
def add_signal():
    signal = request.json
    with open(SIGNALS_FILE, 'r+') as f:
        signals = json.load(f)
        signals.append(signal)
        f.seek(0)
        json.dump(signals, f)
    return jsonify({'status': 'success'})

@app.route('/add_currency', methods=['POST'])
def add_currency():
    currency = request.json
    with open(CURRENCIES_FILE, 'r+') as f:
        currencies = json.load(f)
        currencies.append(currency)
        f.seek(0)
        json.dump(currencies, f)
    return jsonify({'status': 'success'})

@app.route('/add_indicator', methods=['POST'])
def add_indicator():
    indicator = request.json
    with open(INDICATORS_FILE, 'r+') as f:
        indicators = json.load(f)
        indicators.append(indicator)
        f.seek(0)
        json.dump(indicators, f)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)