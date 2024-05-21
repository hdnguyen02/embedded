import os

current_script_path = os.path.abspath(__file__)
project_directory = os.path.dirname(current_script_path)
file_path = project_directory + os.path.normpath( "\\id_staff.txt")


def read_user_id():
    global file_path
    with open(file_path, 'r') as file:
        user_id = int(file.read().strip())
        return user_id


def write_user_id(user_id):
    global file_path
    with open(file_path, 'w') as file:
        file.write(str(user_id))


def generate_user_id():
    last_user_id = read_user_id()
    new_user_id = last_user_id + 1
    write_user_id(new_user_id)
    return new_user_id
