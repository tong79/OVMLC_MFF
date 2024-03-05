import pandas as pd
import pickle
import tensorflow as tf
test_annota_csv = pd.read_csv('datasets/OpenImages/2018_04/test/test-annotations-human.csv')
test_images_csv = pd.read_csv('datasets/OpenImages/2018_04/test/images.csv')['ImageID']
def read_pickle(work_path):
    data_list = []
    with open(work_path, "rb") as f:
            while True:
                try:
                    data = pickle.load(f)
                    data_list.append(data)
                except EOFError:
                    break
    return data_list

path = 'datasets/OpenImages/2018_04/'
seen_labelmap_path = path+'/classes-trainable.txt'
dict_path = path+'/class-descriptions.csv'
unseen_labelmap_path = path+'/unseen_labels.pkl'
seen_labelmap, unseen_labelmap, label_dict = LoadLabelMap(seen_labelmap_path, unseen_labelmap_path, dict_path)
pkl_path = path + 'unseen_labels.pkl'
data_list = read_pickle(pkl_path)
print(data_list)
