import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[10,20,25,30,40]

plt.plot(x,y,marker='o')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Basic Line Plot")
#plt.show()
plt.savefig('line-plot.png')