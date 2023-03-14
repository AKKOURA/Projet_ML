from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)

#Cela permettra à l' API back de recevoir des requêtes de front
#CORS(app, resources={r"/http://127.0.0.1:5000/": {"origins": "http://localhost:4200"}})
@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

@app.route('/hello')
def hello():
    return jsonify({'message': 'Hello, World!'})

if __name__ == '__main__':
    app.run(debug=True)