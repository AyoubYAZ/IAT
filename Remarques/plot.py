# import pandas
import csv
import matplotlib.pyplot as plt

with open('Plot.csv', 'w', newline='') as file:
    writer = csv.writer(file)
     
    writer.writerow(["Episodes", "Score", "Q sum"])
    x,y,r=[],[],[]
    for i in [line.rstrip('\n') for line in open("Q_E10000_S500_G0.9_I1_F0.001.csv","r")][1:]:
        i=i.split(",")
        x.append(int(i[0]))
        r.append(int(float(i[2])))
        y.append(round(float(i[-1]),2))
        

    plt.plot(x,y)
    plt.xlabel('Episodes')
    plt.ylabel('Sum_Q')
    plt.show()