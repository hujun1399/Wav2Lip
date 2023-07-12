import time
from flask import Flask, jsonify, request

from infer_charactor_server import InferCharactorModel

app = Flask(__name__)
model = InferCharactorModel()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        start_time = time.time()
        user_data = request.get_json()
        identity_name = user_data["identity_name"]
        audio_file = user_data["audio_file"]
        video_name = user_data["video_name"]
        print(user_data)
        model.inference(identity_name, audio_file, video_name)
        # model.inference("kanghui", audio_file="./results/source_9s.mp3", video_name="test_kanghui.mp4")
        end_time = time.time()
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>total cost time: ", end_time - start_time)
        return jsonify({'cost time': end_time - start_time})


if __name__ == '__main__':
    app.run()