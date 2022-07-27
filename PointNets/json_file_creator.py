from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import glob
import json

#file_list1 = glob.glob('lidar_dataset/02691156/points/*.pts')
#file_list2 = glob.glob('lidar_dataset/02773838/points/*.pts')
#file_list3 = glob.glob('lidar_dataset/02954340/points/*.pts')
##file_list4 = glob.glob('lidar_dataset/02958343/points/*.pts')
##file_list5 = glob.glob('lidar_dataset/03001627/points/*.pts')
#file_list6 = glob.glob('lidar_dataset/03261776/points/*.pts')
#file_list7 = glob.glob('lidar_dataset/03467517/points/*.pts')

#file_list1 = glob.glob('lidar_dataset5/02691156/points/*.pts')
#file_list2 = glob.glob('lidar_dataset5/02773838/points/*.pts')
#file_list3 = glob.glob('lidar_dataset5/02954340/points/*.pts')
#file_list4 = glob.glob('lidar_dataset5/03467517/points/*.pts')

#file_list1 = glob.glob('lidar_dataset3/02691156/points/*.pts')
#file_list2 = glob.glob('lidar_dataset3/02773838/points/*.pts')
#file_list3 = glob.glob('lidar_dataset3/02954340/points/*.pts')
#file_list4 = glob.glob('lidar_dataset3/03467517/points/*.pts')
#file_list5 = glob.glob('lidar_dataset3/99999999/points/*.pts')

# inlier outlier
#file_list1 = glob.glob('filter_dataset/02691156/points/*.pts')
#file_list2 = glob.glob('filter_dataset/02773838/points/*.pts')

# inlier
#file_list1 = glob.glob('filter_dataset/02691156/points/*.pts')

# outlier
file_list1 = glob.glob('kittitestood_dataset/02773838/points/*.pts')

# inlier car, outliers
#file_list1 = glob.glob('filter_dataset3/02691156/points/*.pts')
#file_list2 = glob.glob('filter_dataset3/99999999/points/*.pts')

# bbox dataset
#file_list1 = glob.glob('kittitest_dataset/02691156/points/*.pts')
#file_list2 = glob.glob('kittitest_dataset/02773838/points/*.pts')
#file_list3 = glob.glob('kittitest_dataset/02954340/points/*.pts')

# near dataset
#file_list1 = glob.glob('near_ID_dataset_newbox_moredata/02691156/points/*.pts')
#file_list2 = glob.glob('near_ID_dataset_newbox_moredata/02773838/points/*.pts')
#file_list3 = glob.glob('near_ID_dataset_newbox_moredata/02954340/points/*.pts')

#file_list = file_list1 + file_list2 + file_list3 + file_list4 + file_list5 + file_list6 + file_list7
#file_list = file_list1 + file_list2 + file_list3 + file_list6 + file_list7
#file_list = file_list1 + file_list2 + file_list3 + file_list4 + file_list5
#file_list = file_list1 + file_list2 + file_list3 + file_list4
#file_list = file_list1 + file_list2 + file_list3
#file_list = file_list1 + file_list2
file_list = file_list1

#print(len(file_list))

#file_list = [1,2,5,3,5,7,4,5,3,8]

#print(file_list)

# default test size = 0.2
train, test = train_test_split(file_list, test_size=0.2, random_state=42)
print(len(train)) # 25343 = size of inlier dataset
print(len(test))

#file_shuffle = shuffle(file_list, random_state=None)

with open('shuffled_train_file_list.json', 'w') as f:
    json.dump(file_list, f)

with open('shuffled_test_file_list.json', 'w') as f:
    json.dump(file_list, f)