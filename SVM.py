import numpy as np
import cv2
from matplotlib import pyplot as plt

CIFAR_DIR = "/users/hashu/Desktop/Computer_Vision/SVM_Image_Classification/cifar-10-batches-py/"

dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4', 'data_batch_5', 'test_batch']

all_data = [0,1,2,3,4,5,6]

def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        cifar_dict = pickle.load(fo)
    return cifar_dict

for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

batch_meta =  all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

label_names = batch_meta["label_names"]
#print(label_names)
##getting an image from data_batch1
#print(image1.shape) #10000 pics of 3, 32, 32....but opencv needs 32x32x3
#number_of_images = 3 #10,0000 takes 22.641 secs




def create_input_maxtrix(number_of_images,index,databatch): #databatch(int) related to which batch to use

    start = index*number_of_images
    end = index*number_of_images + number_of_images
    matx = []

    if databatch==1:
        image1 = data_batch1["data"]
        labels1 = data_batch1["labels"]

    elif databatch==2:
        image1 = data_batch2["data"]
        labels1 = data_batch2["labels"]

    elif databatch==3:
        image1 = data_batch3["data"]
        labels1 = data_batch3["labels"]

    elif databatch==4:
        image1 = data_batch4["data"]
        labels1 = data_batch4["labels"]

    else:# databatch==5
        image1 = data_batch5["data"]
        labels1 = data_batch5["labels"]


        image1 = np.reshape(image1,(10000,3,32,32))
        labels1 = np.array(labels1)


    for i in range(number_of_images):
        tester = image1[i].transpose(1,2,0)
        #array is RGB, cv2 requires BGR
        img = cv2.cvtColor(tester,cv2.COLOR_RGB2BGR)
        #print(labels1.shape)
        #name = labels1[i]
        #print(label_names[name])
        plt.imshow(img, interpolation = 'nearest')
        #plt.show()
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image.ravel()
        myarray = gray_image[0].astype('int8')
        matx.append(gray_image)

    matx = np.asarray(matx)
    matx = np.transpose(matx)
    #print(matx) #rc is number of images by number of pixels per image
    #cv2.imshow('grey',gray_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()]
    trueLabels = labels1[start:end]
    return matx,trueLabels


#support vector machine
def svm(X,Y):

    w=np.zeros(len(X[0]))

    eta = 1 #learning rate

    epochs = 100 #number of iterations

    errors = []

    for epoch in range(1, epochs):
        error = 0
        for i,x in enumerate(X):

            if (Y[i]*np.dot(X[i],w)) < 1:

                w = w + eta * ((X[i] * Y[i]) + (-2 * (1/epoch)* w) )
                error = 1

            else:

                w = w + eta * (-2* (1/epouch)* w)

        errors.append(error)

    #ploting the error

    plt.plot(error,'|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

    return w

def lossfunc(X,Y,W):
    delta =1
    req =1 #regularization value , used to discoverage complex hyperplanes
    dW = np.zeros(W.shape) #init the gradient as zero
    #first step is to calcuate the score
    num_classes = W.shape[0] #number of classes
    num_train = X.shape[1] #number of images to be trained
    loss = 0.0
    #each image is an input for training
    for i in range(num_train):
        scores = W.dot(X[:,i])
        CorrectClassScore = scores[Y[i]]
        for j in range(num_classes):
            if j == Y[i]:
                continue
            margin = np.maximum(0,scores[j] - CorrectClassScore + delta)
            loss +=margin
            if margin >0:
                dW[j:] +=X[:,i].T #
                dW[Y[i],:] -= X[:,i].T
    loss =loss/ num_train #average loss
    dW = dW/num_train #average gradient

    loss += 0.5*req*np.sum(W*W) #regularization loss

    dW += req*W #regularization

    #print(loss)

    #print(dW)

    return loss,dW

    #print(score - score[y])
    #print(score.shape)
    #need to remove the score for the correct class
    ##margins = np.maximum(0, score - score[y] + delta)
    ##margins[y] = 0
    ##loss = np.sum(margins)
    ##return loss



def main():
    #create a random w (10 by 32*32)
    W = np.random.rand(10,1024)
    number_of_images = 5; #images per training set

    #matx,trueLabels = create_input_maxtrix(number_of_images,i)
    #print(trueLabels) #matx is 3 rows and 1024 columns

    #loss,dW = lossfunc(matx,trueLabels,W)
    print(W)
    #print(W- dW)
    losses = np.array([])
    for j in range(5):

        for i in range(1):
            print(i)
            print(j)
            matx,trueLabels = create_input_maxtrix(number_of_images,0,0)
            loss,dW = lossfunc(matx,trueLabels,W)
            W += -0.000005*dW
            losses = np.append(losses,loss)
    #print(W)
    #print(loss)
    print(losses)




    #print(len(matx)) #matx is 3 rows and 1024 columns
    #print(len(matx[0]))


if __name__ =='__main__':
    main()
