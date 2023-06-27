from flask import Flask, jsonify, request, render_template
from PIL import Image, ImageOps, ImageFilter
import base64
import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings

from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore", message="This is a development server. Do not use it in a production deployment.")

app = Flask(__name__)

# 加载训练好的模型
model = tf.keras.models.load_model('model/model1.h5')

print(model.summary())
print(model.input_shape)
print(model.output_shape)


@app.route('/')
def index():
    return render_template('index.html')


# 路由处理程序
@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获取图像数据
    data = request.get_data()
    data = str(data)
    data = data.split(",")[1]
    image_data = base64.b64decode(data)
    # 将图像数据转换为图像对象

    try:
        # 将Base64解码后的数据包装成 BytesIO 对象
        image_stream = io.BytesIO(image_data)
        # Image.open(image_stream).show()
        # Image.open(image_stream).save('test.png')
        # 将 BytesIO 对象传递给 Image.open() 函数
        image = Image.open(image_stream).convert('L')
        # image.show()
        # image.save('test1.png')
    except Image.UnidentifiedImageError as e:
        print(e)
        return jsonify({'error': '无法识别的图像格式'})

    # 对图像进行预处理
    image = np.array(image.resize((28, 28)))
    plt.imshow(image, cmap='gray')
    plt.show()
    # plt.imsave('test2.png', image, cmap='gray')
    image = image.reshape((1, 28, 28, 1))
    image = image / 255.0
    # 进行预测
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    preds = model.predict(image)[0]
    prediction = model.predict(image).argmax()
    results = []
    for i, prob in enumerate(preds):
        result = {'value': round(prob * 100, 2), 'name': class_names[i]}
        results.append(result)
    print(prediction)
    print(results)
    image_stream.close()
    return jsonify({'result': str(int(prediction)), 'pred': results})


@app.route('/upload', methods=['POST'])
def upload():
    # 从请求中获取图像数据
    # data = request.get_data()
    # data = str(data)
    json_data = request.get_json()
    data = json_data['image']
    # data = data.split(",")[1]
    image_data = base64.b64decode(data)
    # 将图像数据转换为图像对象
    try:
        # 将Base64解码后的数据包装成 BytesIO 对象
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        # image.show()
        print(image.size)
        width, height = image.size

        new_width = int(width * 1)
        new_height = int(height * 1)

        image = image.resize((new_width, new_height), Image.ANTIALIAS)

        x = (new_width - width) // 2
        y = (new_height - height) // 2
        crop_width = width
        crop_height = height

        image = image.crop((x, y, x + crop_width, y + crop_height))
        # image.show()
        # if width > 300 and height > 300:
        #     left = (width - 300) / 2
        #     top = (height - 300) / 2
        #     right = (width + 300) / 2
        #     bottom = (height + 300) / 2
        #
        #     image = image.crop((left, top, right, bottom))
        #
        # # 将 BytesIO 对象传递给 Image.open() 函数
        image = image.convert('L')
        print(image.size)


    except Image.UnidentifiedImageError as e:
        print(e)
        return jsonify({'error': '无法识别的图像格式'})
    # 对图像进行预处理
    image = np.array(image.resize((28, 28)))
    plt.imshow(image, cmap='gray')
    plt.show()
    image = image.reshape((1, 28, 28, 1))
    image = image / 255.0

    # 进行预测
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    preds = model.predict(image)[0]
    prediction = model.predict(image).argmax()
    results = []
    for i, prob in enumerate(preds):
        result = {'value': round(prob * 100, 2), 'name': class_names[i]}
        results.append(result)
    print(prediction)
    print(results)
    image_stream.close()
    return jsonify({'result': str(int(prediction)), 'pred': results})

    #     # 获取上传的文件
    #     file = request.files['file']
    #
    #     # 保存文件到本地
    #     filename = secure_filename(file.filename)
    #     file.save(filename)
    #
    #     # 读取图像数据
    #     image_rotate = Image.open(filename)
    #     image_rotate.show()
    #     image = image_rotate.convert('L')
    #     image.show()
    #     threshold = 128  # 二值化阈值
    #     image = image.point(lambda x: 0 if x < threshold else 255, '1')
    #
    #     image.show()
    #     # 对图像进行预处理
    #     image = np.array(image.resize((28, 28)))
    #     image = image.reshape((1, 28, 28, 1))
    #     image = image / 255.0
    #     prediction = model.predict(image).argmax()
    #     print(prediction)
    #     # 返回预测结果
    #     os.remove(filename)
    #     return jsonify({'result': str(int(prediction))})
    #

    # 处理 favicon.ico 请求
    @app.route('/favicon.ico')
    def favicon():
        return ""

    if __name__ == '__main__':
        app.run(debug=True)
