import os


def normalize_yolo(a):
    image_width = 480
    image_height = 360
    return str(a[0]) + " " + str((a[1]+a[3]/2)/image_width) + " " + str((a[2]+a[4]/2)/image_height) + " " + str(a[3]/image_width) + " " + str(a[4]/image_height) + " "


def normalize_chunks(from_index, to_index, from_file, to_file):
    for name in range(from_index, to_index):
        name_string = "000" + str(name)
        name_string = name_string[-4:]
        # name_string_second = ""
        # if to_file == "validation":
        #     name_second = name - from_index + 1
        #     name_string_second = "000" + str(name_second)
        #     name_string_second = name_string_second[-4:]
        # else:
        #     name_string_second = name_string

        with open(os.getcwd() + "/data/ears/annotations/detection/"+from_file+"_YOLO_format/" + name_string + ".txt", 'r') as read_file, open(os.getcwd() + "/data/ears/annotations/detection/true_"+to_file+"_YOLO_format/" + name_string + ".txt", "w+") as write_file:
            lines = read_file.readlines()
            for line in lines:
                annotation = list(map(int, line.split()))
                write_file.write(normalize_yolo(annotation)+"\n")



def normalize_data():
    # normalize test
    normalize_chunks(1, 251, "test", "test")

    # normalize train and make last 100 validation set
    normalize_chunks(1, 651, "train", "train")
    normalize_chunks(651, 751, "train", "validation")


if __name__ == '__main__':
    print(os.getcwd())
    normalize_data()
