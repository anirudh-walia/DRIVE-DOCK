from flask import Flask, render_template, Response, request, redirect
import os
import sqlite3
from werkzeug.utils import secure_filename
from camera import Video
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)
conn = sqlite3.connect('minorprojdb.sqlite', check_same_thread=False)

cursor = conn.cursor()  # cursors for the database
cursor2 = conn.cursor()

@app.route('/')  # loads the landing page
def index():
    return render_template('DriveDock.html')


@app.route('/signup')  # signup page
def signup():
    return render_template('signUp.html')


@app.route('/signin')  # signin page
def signin():
    return render_template('signin.html')


@app.route('/saveuserdata', methods=['POST'])
def saveuserdata():
    firstname = request.form.get('fname')
    lastname = request.form.get('lname')
    email = request.form.get('email')
    password = request.form.get('password')
    msg = "USER REGISTERED SUCCESSFULLY"

    cursor.execute(
        """INSERT INTO `user_data` (`firstname`, `lastname`, `email`, `password`) VALUES ('{}','{}','{}','{}')""".format(
            firstname, lastname, email, password)
    )
    conn.commit()
    return render_template('signin.html', msg=msg)


@app.route('/signinverification', methods=['POST'])
def verify():
    email = request.form.get('email')
    password = request.form.get('password')
    cursor.execute(
        """SELECT * FROM `user_data` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(email, password))
    users = cursor.fetchall()
    if len(users) > 0:
        return redirect('/home')
    else:
        return redirect('/signin')


@app.route('/home')
def home():
    cursor.execute(
        """SELECT * FROM `drivers` """)
    data = cursor.fetchall()
    data = np.array(data).flatten()
    cursor2.execute(
        """SELECT * FROM `dd_values` """)
    data2 = cursor2.fetchall()
    data2 = np.array(data2).flatten()
    print(data2)
    return render_template('cars.html', data=data, data2=data2)


@app.route('/videoopen', methods=['POST'])
def videoopen():
    return render_template('video.html')


def gen(camera):
    while True:
        counter = 0
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type:  image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


@app.route('/video')
def video():
    return Response(gen(Video()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/savedriver', methods=['POST'])
def savedriver():
    driver = request.form.get('driver')
    cursor.execute(
        """INSERT INTO `drivers` (`driver`) VALUES ('{}')""".format(driver))
    conn.commit()
    return redirect('/home')


@app.route('/getdata', methods=['GET'])
def get_data():
    status = request.args.get('status')
    print(status)
    cursor.execute(
        """INSERT INTO `dd_values` (`status`) VALUES ('{}')""".format(status))
    conn.commit()
    return status


if __name__ == '__main__':
    app.run(debug=True, host='192.168.111.1')
