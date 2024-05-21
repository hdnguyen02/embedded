import os

from firebase_admin import credentials, initialize_app, firestore, storage
from flask import Flask, session, render_template, request, Response, redirect
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_query import BaseCompositeFilter
from datetime import timedelta, datetime
import cv2
from flask_socketio import SocketIO
from multiprocessing import Process
import shutil

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread, Event

from AI.model import create_1D_neural_network
from AI.helpers import bgr_2_grayscale, load_dataset_from_directory, hex_to_str_array

from common import generate_user_id

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
socketio = SocketIO(app)

json_path = os.path.join(app.root_path, 'static', 'config', 'firebase_credentials.json')
cred = credentials.Certificate(json_path)
initialize_app(cred, {
    'storageBucket': 'embedded-2fcfe.appspot.com'
})
bucket = storage.bucket()
db = firestore.client()
users_ref = db.collection('users')

is_capture = False
uid = ''
faces = []

plot_event = Event()
hist = None


@app.route('/')
def show_login():
    if is_authenticate():
        return redirect('/manage-staffs')
    return render_template('login.html')


@app.route('/log-out')
def log_out():
    session.pop('authenticated', None)
    session.pop('email', None)
    session.pop('uid', None)
    session.pop('firstName', None)
    session.pop('lastName', None)
    return redirect('/')


@app.route('/', methods=['POST'])
def handle_login():
    form = request.form
    email = form['email']
    password = form['password']
    query = users_ref.where(filter=BaseCompositeFilter("AND", [FieldFilter('role', '==', 'admin'),
                                                               FieldFilter('email', '==', email),
                                                               FieldFilter('password', '==', password)]))
    results = query.stream()
    data = [{**result.to_dict(), 'id': result.id} for result in results]
    if len(data) > 0:
        session['authenticated'] = True
        session['email'] = email
        session['uid'] = data[0]['id']
        session['firstName'] = data[0]['firstName']
        session['lastName'] = data[0]['lastName']
        return redirect('/manage-staffs')


def is_authenticate():
    if session.get('authenticated'):
        return True
    return False


@app.route('/manage-staffs')
def show_manage_staffs():
    if not is_authenticate():
        return render_template('not-authenticate.html')

    query = users_ref.where('role', '==', 'staff').get()
    staffs = []
    for doc in query:
        staff = doc.to_dict()
        staffs.append(staff)
    return render_template('manage-staffs.html', staffs=staffs)


@app.route('/manage-staffs/create')
def show_create_staffs():
    if not is_authenticate():
        return render_template('not-authenticate.html')
    return render_template('create-staff.html')


@app.route('/manage-staffs/create', methods=['POST'])
def create_staffs():
    if not is_authenticate():
        return render_template('not-authenticate.html')
    form = request.form

    file = request.files['avatar']  # lưu lên firestore.
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')
    ext = os.path.splitext(file.filename)[1]
    new_filename = f'{timestamp}{ext}'

    blob = bucket.blob(f'avatars/{new_filename}')
    blob.upload_from_file(file.stream)
    blob.make_public()
    image_url = blob.public_url
    # khởi tạo id user.
    id_staff = generate_user_id()

    data = {
        'email': form['email'],
        'password': form['password'],
        'firstName': form['firstName'],
        'lastName': form['lastName'],
        'role': 'staff',
        'phone': form['phone'],
        'gender': form['gender'],
        'dateOfBirth': form['dateOfBirth'],
        'enable': True,
        'imageUrl': image_url,
        'id': id_staff
    }
    users_ref.add(data)
    return redirect('/manage-staffs')


@app.route('/register-face/<int:id_staff>', methods=['GET'])
def get_register_face(id_staff):
    global uid
    if not is_authenticate():
        return render_template('not-authenticate.html')
    uid = id_staff
    return render_template('register-face.html')


def get_face(frame):
    casc_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + casc_path)
    facesx = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    if len(facesx) > 0:
        (x, y, w, h) = facesx[0]
        face = frame[y + 2:y + h - 2, x + 2:x + w - 2]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame, face
    return frame, []


def save_image(frame, id_user):
    image_user_dir = os.path.join(os.getcwd(), 'static', 'image', str(id_user))
    if not os.path.exists(image_user_dir):
        os.makedirs(image_user_dir)
    current_time = datetime.now().strftime('%Y%m%d%H%M%S%f')
    file_name = f'image_{current_time}.jpg'
    file_path = os.path.join(image_user_dir, file_name)
    cv2.imwrite(file_path, frame)


def save_image_concurrently(frame, id_user):
    process = Process(target=save_image, args=(frame, id_user))
    process.start()


def generate_frames():
    global is_capture
    global uid
    camera = cv2.VideoCapture(0)  # Mở camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, face = get_face(frame)
            if len(face) != 0 and is_capture:
                faces.append(face)
                socketio.emit("showTotalCapture", {'totalDataset': len(faces)})

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('capture')
def handle_capture():
    global is_capture
    global faces
    faces = []
    is_capture = True


@socketio.on('stop-capture')
def handle_stop_capture():
    global is_capture
    global uid
    global faces
    is_capture = False
    for face in faces:
        save_image_concurrently(face, uid)
    faces = []


@app.route('/dataset/<string:id_user>')
def get_dataset(id_user):
    if not is_authenticate():
        return render_template('not-authenticate.html')

    dataset_dir = os.path.join(app.root_path, 'static', 'dataset', id_user)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
    dataset_urls = [os.path.join('\\static', 'dataset', id_user, f) for f in dataset_files]
    return render_template('dataset.html', dataset_urls=dataset_urls)


@app.route('/dataset/<string:id_user>/create')
def get_create_dataset(id_user):
    if not is_authenticate():
        return render_template('not-authenticate.html')
    image_dir = os.path.join(app.root_path, 'static', 'image', id_user)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    image_urls = [os.path.join('\\static', 'image', id_user, f) for f in image_files]
    return render_template('create-dataset.html', id_user=id_user, image_urls=image_urls)


@app.route('/dataset/<string:id_user>/create', methods=['POST'])
def create_dataset(id_user):
    if not is_authenticate():
        return render_template('not-authenticate.html')

    image_urls = request.form.getlist('image-urls')
    current_script_path = os.path.abspath(__file__)
    project_directory = os.path.dirname(current_script_path)
    destination_directory = project_directory + os.path.normpath("\\static\\dataset\\" + id_user)
    for image_url in image_urls:
        source_image = project_directory + os.path.normpath(image_url)
        try:
            shutil.copy(source_image, destination_directory)
        except FileNotFoundError:
            print("Not found file")
    return redirect(f'/dataset/{id_user}')


@socketio.on('deleteDatasets')
def handle_delete_dataset(data):
    current_script_path = os.path.abspath(__file__)
    project_directory = os.path.dirname(current_script_path)
    dataset_urls = data['datasetUrls']
    for url in dataset_urls:

        normalized_path = project_directory + os.path.normpath(url)
        try:
            os.remove(normalized_path)
        except FileNotFoundError:
            print(f"not found: ${url}")
    socketio.emit('doneDeleteDataset')


@socketio.on('deleteImages')
def handle_delete_image(data):
    current_script_path = os.path.abspath(__file__)
    project_directory = os.path.dirname(current_script_path)
    image_urls = data['imageUrls']
    for url in image_urls:
        normalized_path = project_directory + os.path.normpath(url)
        try:
            os.remove(normalized_path)
        except FileNotFoundError:
            print(f"not found: ${url}")
    socketio.emit('doneDeleteImage')


# hiệu chỉnh nhân viên => có thể vô hiệu hoá nhân viên.
@app.route('/edit/<int:id_user>')
def get_edit_user(id_user):
    if not is_authenticate():
        return render_template('not-authenticate.html')
    docs = users_ref.where('id', '==', id_user).get()
    staff = docs[0].to_dict()
    return render_template('edit-staff.html', staff=staff)


@app.route('/edit/<int:id_user>', methods=['POST'])
def edit_user(id_user):
    if not is_authenticate():
        return render_template('not-authenticate.html')
    form = request.form
    docs = users_ref.where('id', '==', id_user).get()
    id_doc = docs[0].id
    staff = docs[0].to_dict()
    staff['enable'] = True if form['enable'] == 'True' else False
    staff['password'] = form['password']
    users_ref.document(id_doc).update(staff)
    return redirect('/manage-staffs')

@app.route('/manage-models')
def manage_models():
    if not is_authenticate():
        return render_template('not-authenticate.html')
    
    models = []
    model_docs = db.collection("models").stream()
    for doc in model_docs:
        model = doc.to_dict()
        # Chuyển đổi string sang datetime
        model["time"] = datetime.strptime(model['time'], "%d-%m-%Y %H:%M:%S")
        model["mid"] = doc.id
        models.append(model)
    # Sắp xếp theo time giảm dần
    sorted_models = sorted(models, key = lambda x: x['time'], reverse=True)
    # Format lại prop time
    for model in sorted_models:
        model["time"] = model["time"].strftime("%d-%m-%Y %H:%M:%S")
    return render_template('manage-models.html', models = sorted_models)

@app.route('/manage-models/change-model/<string:model_id>')
def change_model(model_id):
    if not is_authenticate():
        return render_template('not-authenticate.html')
    
    models_ref = db.collection("models")

    # Thay đổi isSelected của các model từ True sang False
    selected_model_docs = models_ref.where("isSelected", "==", True).stream()
    for doc in selected_model_docs:
        models_ref.document(doc.id).update({"isSelected": False})

    # Cập nhật isSelected của model yêu cầu sang True
    final_model = models_ref.document(model_id).get()
    if final_model.exists:
        models_ref.document(model_id).update({"isSelected": True, "isEmbedded":False})

    return redirect("/manage-models")

@app.route('/manage-models/train-model')
def train_model():
    if not is_authenticate():
        return render_template('not-authenticate.html')
    
    return render_template('train-model.html')

@socketio.on("trainNewModel")
def train_new_model():
    print("Start training")
    dataset_path = "./static/dataset"
    # Tính số nhãn model cần phần loại (số nhãn = int(nhãn lớn nhất) + 1)
    # Do model yêu cầu label đánh số từ [0,số nhãn dataset) nên để tiện ánh xạ với user id, đặt số nhãn cần phân loại là nhãn lớn nhất + 1 (bao gồm nhãn unknown - 0)
    num_classes = 0
    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        # Kiểm tra xem subdir có phải là thư mục không
        if os.path.isdir(subdir_path):
            if num_classes < int(subdir): 
                num_classes = int(subdir)
    if num_classes == 0:
        socketio.emit("noDataset", {"msg": "Không có dataset để train"})
        return
    num_classes += 1

    image_width = 20 
    image_height = 20
    batch_size = 32
    # Load data từ thư mục dataset
    images, y_train = load_dataset_from_directory(directory=dataset_path, shuffle=True, shape=(image_width, image_height))

    # Chuyến sang GRAYSCALE
    gray_images = []
    for image in images:
        gray_images.append(bgr_2_grayscale(image))
    # Chuyển sang 1D
    x_train = []
    for image in gray_images:
        x_train.append(image.flatten())
    x_train = np.array(x_train)
    print("Finish loading dataset")
    # Khởi tạo model và một số hàm kích hoạt
    model = create_1D_neural_network(lenght=image_width*image_height, class_num=num_classes)
    model.compile(loss='sparse_categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 20, 
                                        restore_best_weights = True)
    print("Start training model")
    # Train
    global hist
    hist = model.fit(x_train, 
                 y_train,
            epochs=100,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[earlystopping])
    print("Finish training model")

    model_path = os.path.join("static", "model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Lưu sơ đồ đánh giá
    thread = Thread(target=save_plot)
    thread.start()
    plot_event.wait()

    # Lưu model
    model.save(os.path.join("static/model", "model.h5"))

    socketio.emit("finishTraining")

def save_plot():
    global hist
    # plt.ion()
    # Biểu đồ đánh giá loss
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label="lost")
    plt.plot(hist.history['val_loss'], color='orange', label="val_loss")
    fig.suptitle("loss", fontsize=20)
    plt.legend(loc = "upper left")
    plt.savefig('./static/model/loss_plot.png')
    plt.close(fig)
    # Biểu đồ đánhg giá accuracy
    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label="accuracy")
    plt.plot(hist.history['val_accuracy'], color='orange', label="val_accuracy")
    fig.suptitle("accuracy", fontsize=20)
    plt.legend(loc = "upper left")
    plt.savefig('./static/model/accuracy_plot.png')
    plt.close(fig)

    # plt.ioff()
    plot_event.set()


@socketio.on("saveCurrentModel")
def save_current_model(data):
    print("Start saving")

    model_document = {
        "decs": data["decs"],
        "isEmbedded": False,
        "isSelected": False,
        "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "fileUrl" : ""
    }
    _, ref = db.collection("models").add(model_document)
    # Lưu file model
    blob = bucket.blob(f"models/{ref.id}.h5")
    blob.upload_from_filename(os.path.join("static/model", "model.h5"))
    file_url = blob.public_url
    # Cập nhật dowload url cho model_document
    db.collection("models").document(ref.id).update({"fileUrl" : file_url})

    socketio.emit("finishSaving")

if __name__ == '__main__':
    app.run()
