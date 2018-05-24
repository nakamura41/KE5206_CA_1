import os, shutil


def add_count(key, dict):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1
    return dict


def data_generate(org_dir, dest_dir, num_of_each_class, sep_by_class=1):
    origin_count = {}
    new_count = {}
    fileNames = os.listdir(org_dir)
    for fn in fileNames:
        lable = int(fn.split('.')[0].split('_')[0])
        num = int(fn.split('.')[0].split('_')[1])
        origin_count = add_count(lable, origin_count)
        if num < num_of_each_class:
            new_count = add_count(lable, new_count)
            if sep_by_class:
                if not os.path.exists(dest_dir + str(lable) + '/'):
                    os.makedirs(dest_dir + str(lable) + '/')
                shutil.copyfile(os.path.join(org_dir, fn), dest_dir + str(lable) + '/' + fn)
            else:
                shutil.copyfile(os.path.join(org_dir, fn), os.path.join(dest_dir, fn))
    print(origin_count)
    print(new_count)


# change to your own path
def data_prep():
    root_path = "/Users/lince/Desktop/CA-CI1/"
    origin_data_path = root_path + "Food-11/"
    origin_train_path = origin_data_path + "training/"
    origin_test_path = origin_data_path + "evaluation/"
    train_data_path = root_path + "data/train/"
    test_data_path = root_path + "data/test/"

    if (not os.path.exists(train_data_path)) and not (os.path.exists(test_data_path)):
        os.makedirs(train_data_path)
        os.makedirs(test_data_path)

    print('===train data generate===')
    data_generate(origin_train_path, train_data_path, 400)
    print('===test data generate===')
    data_generate(origin_test_path, test_data_path, 100, 0)


data_prep()

# output
# ===train data generate===
# {0: 994, 10: 709, 1: 429, 2: 1500, 3: 986, 4: 848, 5: 1325, 6: 440, 7: 280, 8: 855, 9: 1500}
# {0: 400, 10: 400, 1: 400, 2: 400, 3: 400, 4: 400, 5: 400, 6: 400, 7: 280, 8: 400, 9: 400}
# ===test data generate===
# {0: 368, 10: 231, 1: 148, 2: 500, 3: 335, 4: 287, 5: 432, 6: 147, 7: 96, 8: 303, 9: 500}
# {0: 100, 10: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 96, 8: 100, 9: 100}
