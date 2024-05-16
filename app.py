import os
from firebase_admin import credentials, initialize_app, firestore
from flask import Flask, session, render_template, request, url_for, redirect
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_query import BaseCompositeFilter
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

json_path = os.path.join(app.root_path, 'static', 'config', 'firebase_credentials.json')
cred = credentials.Certificate(json_path)
initialize_app(cred)
db = firestore.client()
users_ref = db.collection('users')


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
    print(staffs)
    return render_template('manage-staffs.html', staffs=staffs)


@app.route('/manage-staffs/create')
def show_create_staffs():
    if not is_authenticate():
        return render_template('not_authenticate.html')
    return render_template('create_staff.html')


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
    }
    users_ref.add(data)
    return redirect('/manage-staffs')


if __name__ == '__main__':
    app.run()
