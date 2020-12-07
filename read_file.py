import os


def readname():
    # filePath = './exploration_database_and_code/pristine_images'
    filePath = './BSD432_color'
    name = os.listdir(filePath)
    return name, filePath


if __name__ == "__main__":
    name, filePath = readname()
    print(name)
    txt = open("BSD432_test.txt", 'w')
    for i in name:
        # print(filePath + "/" + i)
        # image_dir = os.path.join('./exploration_database_and_code/pristine_images/', str(i))
        image_dir = os.path.join('./BSD432_color', str(i))
        txt.write(image_dir + "\n")

# import os
# import numpy as np
#
# def create_txt(name, path, file_image):
#     txt_path = path + name + '.txt'
#     txt = open(txt_path, 'w')
#     for (i,j) in zip(file_image):
#         image_dir = os.path.join('./dataset/test/', str(i))
#         # label_dir = os.path.join('./dataset/test_label/', str(j))
#         txt.write(image_dir)
#         # txt.write(label_dir)
#
# def read_file(path1):
#     filelist1 = os.listdir(path1)
#     file_image = np.array([file for file in filelist1 if file.endswith('.jpg')], dtype=object)
#     # filelist2 = os.listdir(path2)
#     # file_label = np.array([file for file in filelist2 if file.endswith('.png')], dtype=object)
#     return file_image
#
#
# path1 = './exploration_database_and_code/pristine_images'
# # path2 = './dataset/test_label/'
#
# file_image = read_file(path1)
# create_txt('test', './dataset/', file_image)