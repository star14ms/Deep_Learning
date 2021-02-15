# 리스트를 txt파일로 저장
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

def save_list_to_file(list, file_name):
    if not os.path.isdir('saves'): # os.path.isdir, os.path.isfile()
        createFolder('saves')

    with open(f'saves/{file_name}.txt', 'w') as file:
        file.write(str(list[0]))
        for element in list[1:]:
            file.write(", "+str(element))

def load_list(file_name, trans_int=False, trans_float=True):
    with open(f"saves/{file_name}.txt", 'r') as file:
        if trans_int:
            list_ = list(map(int, file.read().split(',')))
        elif trans_float:
            list_ = list(map(float, file.read().split(',')))
        else:
            list_ = file.read().split(',')
        return list_