import numpy as np
import csv
import ibelief
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATASET = "Bird_10"
SKIPFIRST = True

def __mains__():
    dataset = read_data()

    iteration = read_iteration()

    classes = get_classes(dataset)
    print(classes)

    images = read_features()

    X = []
    y = []
    X_real = []
    y_real = []

    for img in np.unique(dataset[:, 5]):
        answers = np.where(dataset[:,5] == img)[0]
        
        print("\n\n\n\nVraie classe :", dataset[answers[0]][2])

        i = 0
        for user_answer in answers:
            print("Réponse :", i)
            print(dataset[user_answer][4], '  ', dataset[user_answer][3])
            iter = np.where((iteration[:,0] == dataset[user_answer][0]) & (iteration[:,1] == dataset[user_answer][1]))[0]
            if iter.shape[0] >= 1:
                print(iteration[iter[0]][3], '  ', iteration[iter[0]][2])
            i += 1

        #img = mpimg.imread('data/' + DATASET + '/pictures/' + img + '.jpg')
        #plt.imshow(img)
        #plt.show()
        val = int(input("Choix :"))

        bba = get_bba(answers[val], dataset, iteration, classes)

        X.append(images[np.where(images[:,0] == dataset[answers[val]][5])[0]][0, 3:].astype(float))
        X_real.append(dataset[answers[val]][5])
        y.append(bba)
        y_real.append(np.where(classes == dataset[answers[val]][2])[0][0])

    X = np.array(X)
    X_real = np.array(X_real)
    y = np.array(y)
    y_real = np.array(y_real)
    """np.save('data/' + DATASET + '/dataset/X_real', X_real)
    np.save('data/' + DATASET + '/dataset/X', X)
    np.save('data/' + DATASET + '/dataset/y', y)
    np.save('data/' + DATASET + '/dataset/y_real', y_real)
    np.save('data/' + DATASET + '/dataset/classes', classes)"""

def get_bba(val, dataset, iteration, classes):
    bba = np.zeros(2**len(classes))
    answer_weight = 0

    for j in range(len(classes)):
        if str(classes[j]) in str(dataset[val][3]):
                answer_weight += 2**j

    bba[answer_weight] = int(dataset[val][4]) / 7
    bba[2**len(classes) - 1] = 1 - bba[answer_weight]

    iter = np.where((iteration[:,0] == dataset[val][0]) & (iteration[:,1] == dataset[val][1]))[0]
    if iter.shape[0] >= 1:
        bba_iter = np.zeros(2**len(classes))
        answer_weight = 0
        
        for j in range(len(classes)):
            if str(classes[j]) in str(iteration[iter[0]][2]):
                answer_weight += 2**j
                
        bba_iter[answer_weight] = int(iteration[iter[0]][3]) / 7
        bba_iter[2**len(classes) - 1] = 1 - bba_iter[answer_weight]
        
        bba = ibelief.DST(np.array([bba, bba_iter]).T, 9).T[0]
        print(bba)

    return bba

def print_bba(bba, classes):
    vals = np.where(bba != 0)[0]
    for j in vals:
        print("\n",bba[j])
        if(j == 2**len(classes) - 1):
            print("Ignorance")
        else:
            for i in range(len(classes)):
                if j & 2**i:
                    print(classes[i])
    return 0

def read_features():
    images = []
    with open('data/' + DATASET + '/features.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        skipFirst = True

        for row in spamreader:
            if skipFirst:
                skipFirst = False
            else:
                images.append(row)
                
        csvfile.close()
    
    images = np.array(images)
    for image in images:
        image[0] = int(image[0].split("/")[2].split(".")[0])

    return images

def read_data():
    dataset = []
    with open('data/' + DATASET + '/DATA_imperfect.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        skipFirst = SKIPFIRST

        for row in spamreader:
            if skipFirst:
                skipFirst = False
            else:
                data = []
                data.append(row[1]) #user
                data.append(row[2]) #trial
                data.append(row[4]) #goodAnswer
                data.append(row[5]) #Answer
                data.append(row[8]) #Certitude
                data.append(row[3]) #img
                dataset.append(data)
        
        csvfile.close()
    return np.array(dataset)

def read_iteration():
    iterations = []
    with open('data/' + DATASET + '/ITERATION_imperfect.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        skipFirst = SKIPFIRST

        for row in spamreader:
            if skipFirst:
                skipFirst = False
            elif 'NULL' not in row[4]:
                data = []
                data.append(row[1]) #user
                data.append(row[2]) #trial
                data.append(row[3]) #Answer
                data.append(row[4]) #Certitude
                iterations.append(data)
                
        csvfile.close()
    return np.array(iterations)

def get_classes(dataset):
    classes = [row[2] for row in dataset]
    classes = list(set(classes))
    return np.array(classes)

__mains__()
