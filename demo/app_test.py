from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index_test.html')  # Serve the HTML file

@app.route('/process', methods=['POST'])
def process():
    # Get the JSON data sent in the request body
    data = request.get_json()
    filename = data.get('filename', None)

    if filename:
        return jsonify({"message": f"File {filename} received!"}), 200
    else:
        return jsonify({"error": "No filename received!"}), 400


if __name__ == '__main__':
    app.run(debug=True)
