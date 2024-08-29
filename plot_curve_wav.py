import matplotlib.pyplot as plt
import numpy as np



def plot(fpath, epochs):
    Iteration = []
    train_loss = []
    valid_loss = []
    path = fpath + "/train_log.txt"
    with open(path,'r') as file:
        for i, line in enumerate(file.readlines()):
            if i == epochs:
                break
            line = line.strip().split(",")
            #print(line)
            itera,loss = line[0], line[2]
            itera = int(itera.split(':')[1])
            #print(itera)
            loss = loss.split("- ")
            #print(loss)
            trainl = loss[1]
            trainv = loss[2]
            trainl = float(trainl.split(':')[1])
            trainl = "{:f}".format(float(trainl))
            trainv = float(trainv.split(':')[1])
            trainv = "{:f}".format(float(trainv))
            
            #print(trainl)
            #print(type(trainl))
            Iteration.append(itera)
            train_loss.append(trainl)
            valid_loss.append(trainv)
            #print(type(Loss[0]))
            
    train_loss = np.array(train_loss, dtype='float64')
    valid_loss = np.array(valid_loss, dtype='float64')
    # print(type(Loss))
    Iteration= np.array(Iteration)
    plt.title("Loss")
    plt.plot(Iteration, train_loss, color='cyan', label='train loss')
    plt.plot(Iteration, valid_loss,'b', label='valid loss')

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(fpath+"/loss.jpg")

if __name__ == "__main__":
    plot("C:/Users/503503/Desktop", 30)
# plt.show()