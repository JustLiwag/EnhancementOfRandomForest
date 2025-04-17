from flask import Flask, request, jsonify
from predict import predict_sms
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

@app.route('/predict_sms', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('sms', '')
    result = predict_sms(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
