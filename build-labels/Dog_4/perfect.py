import numpy as np
import csv
import ibelief
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATASET = "Dog_4"
SKIPFIRST = True

def main():
    dataset = read_data()

    classes = np.load('data/' + DATASET + '/dataset/classes.npy')
    X_real = np.load('data/' + DATASET + '/dataset/X_real.npy')
    
    print(classes, dataset.shape, X_real.shape)

    X_perfect = np.zeros(X_real.shape[0])

    for i in range(X_real.shape[0]):
        res = np.where(dataset[:,5] == X_real[i])[0]
        for j in range(classes.shape[0]):
            if(classes[j] in dataset[res[0],3]):
                X_perfect[i] = j

    np.save('data/' + DATASET + '/dataset/y_hard', X_perfect)

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

def read_data():
    dataset = []
    with open('data/' + DATASET + '/DATA_perfect.csv', newline='') as csvfile:
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

def get_classes(dataset):
    classes = [row[2] for row in dataset]
    classes = list(set(classes))
    return np.array(classes)

main()
