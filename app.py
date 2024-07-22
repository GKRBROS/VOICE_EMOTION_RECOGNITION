from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from model import load_model, predict_emotion

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'audio_data'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template("index.html")

emotion_mapping = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fear',
    6: 'disgust',
    7: 'surprise'
}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        emotion = predict_emotion(filepath)
        
        # Convert emotion to a string using the emotion_mapping dictionary
        emotion_string = emotion_mapping.get(int(emotion), 'Unknown')
        
        return jsonify({'emotion': emotion_string})
    return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
