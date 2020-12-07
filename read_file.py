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

