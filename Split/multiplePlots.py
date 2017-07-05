import matplotlib.pyplot as plt

def plot1():
    plt.xlabel("x square axis")
    plt.ylabel("y square axis")
    plt.plot([1,2,3,4],[1,4,9,16])
    plt.axis([0,6,0,20])
    plt.savefig("fig1.png")

def plot2():
    plt.clf()
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.plot([1,2,3,4],[1,2,3,4])
    plt.axis([0,5,0,5])
    plt.savefig("fig2.png")

plot1()
plot2()