import os

from firebase_admin import credentials, initialize_app, firestore
from flask import Flask, session, render_template, request, Response, redirect
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_query import BaseCompositeFilter
from datetime import timedelta, datetime
import cv2
from flask_socketio import SocketIO
from multiprocessing import Process
import shutil

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
socketio = SocketIO(app)

json_path = os.path.join(app.root_path, 'static', 'config', 'firebase_credentials.json')
cred = credentials.Certificate(json_path)
initialize_app(cred)
db = firestore.client()
users_ref = db.collection('users')

is_capture = False
uid = ''
faces = []


@app.route('/')
def show_login():
    return render_template('login.html')


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
        return render_template('not_authenticate.html')

    query = users_ref.where('role', '==', 'staff').get()
    staffs = []
    for doc in query:
        staff = doc.to_dict()
        staff['uid'] = doc.id
        staffs.append(staff)
    return render_template('manage-staffs.html', staffs=staffs)


@app.route('/manage-staffs/create')
def show_create_staffs():
    if not is_authenticate():
        return render_template('not_authenticate.html')
    return render_template('create-staff.html')


@app.route('/manage-staffs/create', methods=['POST'])
def create_staffs():
    if not is_authenticate():
        return render_template('not_authenticate.html')
    form = request.form
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
        'numberDataset': 0
    }
    users_ref.add(data)
    return redirect('/manage-staffs')


@app.route('/register-face/<string:id_staff>', methods=['GET'])
def get_register_face(id_staff):
    global uid
    if not is_authenticate():
        return render_template('not_authenticate.html')
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
    image_user_dir = os.path.join(os.getcwd(), 'static', 'image', id_user)
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
    user_ref = users_ref.document(uid)
    doc = user_ref.get()
    user = doc.to_dict()
    total_dataset = user['totalDataset'] + len(faces)
    user['totalDataset'] = total_dataset
    user_ref.update(user)
    faces = []


@app.route('/dataset/<string:id_user>')
def get_dataset(id_user):
    if not is_authenticate():
        return render_template('not_authenticate.html')

    dataset_dir = os.path.join(app.root_path, 'static', 'dataset', id_user)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
    dataset_urls = [os.path.join('\\static', 'dataset', id_user, f) for f in dataset_files]
    return render_template('dataset.html', dataset_urls=dataset_urls)


@app.route('/dataset/<string:id_user>/create')
def get_create_dataset(id_user):
    if not is_authenticate():
        return render_template('not_authenticate.html')
    image_dir = os.path.join(app.root_path, 'static', 'image', id_user)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    image_urls = [os.path.join('\\static', 'image', id_user, f) for f in image_files]
    return render_template('create-dataset.html', id_user=id_user, image_urls=image_urls)


@app.route('/dataset/<string:id_user>/create', methods=['POST'])
def create_dataset(id_user):
    if not is_authenticate():
        return render_template('not_authenticate.html')

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


if __name__ == '__main__':
    app.run()
