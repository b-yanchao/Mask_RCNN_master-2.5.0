import os
path = 'train_dataset/json'  # path为json文件存放的路径
json_file = os.listdir(path)
for file in json_file:
    os.system("python D:/Anaconda3/envs/labelme/Scripts/labelme_json_to_dataset.exe %s"%(path + '/' + file))