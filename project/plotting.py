import matplotlib.pyplot as plt

# Sample data
'''
Plotting the graphs
'''
x=[i for i in range(1,6)]
train_loss=[5.022541300455729,5.122892681757609,5.039763673146566 ,5.029575284322103,5.003616485595703]
val_loss = [4.227301014794244,4.344751781887478,4.207423559824626,4.253160466088189 ,4.09685071309407]
plt.plot(x,train_loss,label='train')
plt.plot(x,val_loss,label='valid')
plt.xlabel('epochs')
plt.ylabel('loss values')
plt.title('GRU attention')
plt.legend() 
plt.show()
