import os, sqlite3, time, base64, io
from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from gtts import gTTS
from model import predict_image
import utils
from nlp import get_nlp_recommendation, extract_crop_from_prediction, extract_disease_from_prediction, get_full_voice_text, get_disease_info_tamil

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# --- DB ---
def get_db():
    c = sqlite3.connect('users.db'); c.row_factory = sqlite3.Row; return c

def init_db():
    db = get_db()
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, filename TEXT NOT NULL, disease TEXT NOT NULL, confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id));
    ''')
    db.commit(); db.close()

init_db()

# --- Helpers ---
def auth(): return 'user_id' in session
def guard(): return None if auth() else redirect(url_for('login'))
def uid(): return session['user_id']
def uname(): return session.get('user_name')

def db_exec(sql, params=()):
    db = get_db(); db.execute(sql, params); db.commit(); db.close()

def db_query(sql, params=()):
    db = get_db(); rows = db.execute(sql, params).fetchall(); db.close(); return rows

def save_img(file_obj, prefix='upload'):
    os.makedirs('static/uploads', exist_ok=True)
    name = f"{int(time.time())}_{prefix}_{secure_filename(getattr(file_obj,'filename','img.jpg'))}"
    path = f"static/uploads/{name}"
    (file_obj.save(path, 'JPEG') if isinstance(file_obj, Image.Image) else file_obj.save(path))
    return name, path

def predict_run(img_bytes, lang='en'):
    import re
    pred, confidence = predict_image(img_bytes)
    name = pred.replace('___', ' - ').replace('_', ' ')
    info = utils.disease_dic[pred]
    crop = extract_crop_from_prediction(pred)
    disease = extract_disease_from_prediction(pred)
    try: rec = get_nlp_recommendation(crop, disease, lang)
    except: rec = f"For {disease} in {crop}: consult an agricultural expert."

    if lang == 'ta':
        ta_info = get_disease_info_tamil(crop, disease)
        disease_info_dict = {'description': '', 'Possible Steps': ta_info}
        ta_plain = re.sub(r'<[^>]+>', ' ', ta_info)
        ta_plain = re.sub(r'\s+', ' ', ta_plain).strip()
        # Voice = recommendation + disease info (no duplicate)
        voice = rec + '. ' + ta_plain
    else:
        disease_info_dict = {'description': '', 'Possible Steps': info}
        info_plain = re.sub(r'<[^>]+>', ' ', info)
        info_plain = re.sub(r'\s+', ' ', info_plain).strip()
        voice = rec + '. ' + info_plain

    return name, info, rec, voice, disease_info_dict, confidence


# --- Auth ---
@app.route('/')
def index(): return redirect(url_for('home') if auth() else url_for('login'))

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        f, e, p = request.form['fullname'], request.form['email'], request.form['password']
        if db_query('SELECT id FROM users WHERE email=?', (e,)):
            flash('Email already registered.', 'danger')
            return render_template('signup.html')
        db_exec('INSERT INTO users(fullname,email,password) VALUES(?,?,?)', (f, e, generate_password_hash(p)))
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        e, p = request.form['email'], request.form['password']
        rows = db_query('SELECT * FROM users WHERE email=?', (e,))
        if rows and check_password_hash(rows[0]['password'], p):
            session.update({'user_id': rows[0]['id'], 'user_name': rows[0]['fullname'], 'user_email': rows[0]['email']})
            return redirect(url_for('home'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('login'))

# --- Simple pages ---
@app.route('/home')
def home():
    if r := guard(): return r
    return render_template('index.html', user_name=uname())

@app.route('/about')
def about():
    if r := guard(): return r
    return render_template('about.html')

@app.route('/camera')
def camera():
    if r := guard(): return r
    return render_template('camera.html', user_name=uname())

@app.route('/model_info')
def model_info():
    if r := guard(): return r
    return render_template('model_info.html')

# --- Predict ---
@app.route('/predict', methods=['GET','POST'])
@app.route('/upload', methods=['GET','POST'])
def predict():
    if r := guard(): return r
    if request.method == 'POST':
        try:
            file = request.files.get('file') or request.files.get('image')
            if not file: flash('No file selected.', 'danger'); return render_template('index.html', user_name=uname())
            lang = request.form.get('language', 'en')
            fname, path = save_img(file)
            clean_name, disease_info, nlp_rec, voice, disease_info_dict, confidence = predict_run(open(path,'rb').read(), lang)
            db_exec('INSERT INTO predictions(user_id,filename,disease,confidence) VALUES(?,?,?,?)', (uid(), fname, clean_name, confidence))
            return render_template('result.html', img=f'/static/uploads/{fname}', disease_name=clean_name,
                result=disease_info, nlp_recommendation=nlp_rec, voice_text=voice, confidence=confidence,
                disease_info=disease_info_dict, language=lang)
        except Exception as e:
            print(e); flash('Error processing image.', 'danger')
    return render_template('index.html', user_name=uname())

@app.route('/capture', methods=['POST'])
def capture():
    if not auth(): return jsonify({'error': 'Not authenticated'}), 401
    try:
        data = request.json
        img = Image.open(io.BytesIO(base64.b64decode(data['image'].split(',')[1])))
        lang = data.get('language', 'en')
        fname, _ = save_img(img, 'camera')
        buf = io.BytesIO(); img.save(buf, 'JPEG')
        clean, info, rec, voice, _, confidence = predict_run(buf.getvalue(), lang)
        db_exec('INSERT INTO predictions(user_id,filename,disease,confidence) VALUES(?,?,?,?)', (uid(), fname, clean, confidence))
        return jsonify({'success': True, 'disease_name': clean, 'confidence': confidence,
            'confidence_warning': '', 'image_path': f'/static/uploads/{fname}',
            'disease_info': str(info), 'nlp_recommendation': rec, 'voice_text': voice})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- History / Dashboard ---
@app.route('/history')
def history():
    if r := guard(): return r
    rows = db_query('SELECT filename,disease,confidence,created_at FROM predictions WHERE user_id=? ORDER BY created_at DESC', (uid(),))
    return render_template('history.html', history=[{'file':r['filename'],'disease':r['disease'],'confidence':r['confidence'],'time':r['created_at']} for r in rows])

@app.route('/dashboard')
def dashboard():
    if r := guard(): return r
    dr = db_query('SELECT disease, COUNT(*) cnt FROM predictions WHERE user_id=? GROUP BY disease ORDER BY cnt DESC LIMIT 10', (uid(),))
    tr = db_query('SELECT DATE(created_at) day, COUNT(*) cnt FROM predictions WHERE user_id=? GROUP BY day ORDER BY day DESC LIMIT 7', (uid(),))
    dl, dc = [r['disease'] for r in dr], [r['cnt'] for r in dr]
    tl, tc = [r['day'] for r in reversed(tr)], [r['cnt'] for r in reversed(tr)]
    return render_template('dashboard.html', disease_labels=dl, disease_counts=dc,
        trend_labels=tl, trend_counts=tc, total=sum(dc), user_name=uname())

# --- Audio ---
@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    if not auth(): return jsonify({'error': 'Not authenticated'}), 401
    try:
        data = request.get_json()
        text, lang = data.get('text',''), data.get('language','en')
        if not text: return jsonify({'error': 'No text'}), 400
        os.makedirs('static/audio', exist_ok=True)
        fname = f"speech_{int(time.time())}.mp3"
        gTTS(text=text, lang='ta' if lang=='ta' else 'en', slow=False).save(f"static/audio/{fname}")
        return jsonify({'success': True, 'audio_url': f'/static/audio/{fname}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
