import numpy as np
import matplotlib.pyplot as plt


'''
omega1 = 1
omega2 = 2
omega3 = 4


t = np.linspace(-10,0,1000)


v1 = omega1*(np.exp(omega1*t))
v2 = omega2*(np.exp(omega2*t))
v3 = omega3*(np.exp(omega3*t))

plt.plot(t,v1 )
plt.plot(t,v2 )
plt.plot(t,v3)
plt.show()


t = np.linspace(0,10,1000)
v1 = omega1 * (np.exp(-omega1*t))
v2 = omega2 * (np.exp(-omega2*t))
v3 = omega3 * (np.exp(-omega3*t))

plt.plot(t,v1)
plt.plot(t,v2)
plt.plot(t,v3)
plt.show()



t = np.linspace(-10,10,100)
omega1 = 1/2
omega2 = 0.6
omega3 = 1
omega4 = 2


v1 = np.sqrt(((omega1**2)/(2*np.pi))*np.exp(((-omega1**2)*(t**2))/2))
v2 = np.sqrt(((omega2**2)/(2*np.pi))*np.exp(((-omega2**2)*(t**2))/2))
v3 = np.sqrt(((omega3**2)/(2*np.pi))*np.exp(((-omega3**2)*(t**2))/2))
v4 = np.sqrt(((omega4**2)/(2*np.pi))*np.exp(((-omega4**2)*(t**2))/2))


plt.plot(t,v1)
plt.plot(t,v2)
plt.plot(t,v3)
plt.plot(t,v4)
plt.show()


omega = 2
gamma = 1
t= np.linspace(-10,0,1000)

v = (4*omega*gamma)/((omega*t*gamma)**2)*np.exp(t*omega)

plt.plot(t,v)
plt.show()
'''
omega = 2 
omega1 = 3
omega2 = 4
omega3 = 6
gamma = 1
t = np.linspace(0,10,100)
v= np.exp(-gamma*t)*((4*omega)/((gamma-omega)**2))*((((np.exp(((-omega+gamma)*t)/2)-1)**2)))
v1 = np.exp(-gamma*t)*((4*omega1)/((gamma-omega1)**2))*((((np.exp(((-omega1+gamma)*t)/2)-1)**2)))
v2 = np.exp(-gamma*t)*((4*omega2)/((gamma-omega2)**2))*((((np.exp(((-omega2+gamma)*t)/2)-1)**2)))
v3 = np.exp(-gamma*t)*((4*omega3)/((gamma-omega3)**2))*((((np.exp(((-omega3+gamma)*t)/2)-1)**2)))
plt.plot(t,v)
plt.plot(t,v1)
plt.plot(t,v2)
plt.plot(t,v3)
plt.show()
'''
gamma

v =  (gamma**2)*(t**2)*np.exp(-gamma*t)
v1 = (gamma1**2)*(t**2)*np.exp(-gamma1*t)
v2 = (gamma2**2)*(t**2)*np.exp(-gamma2*t)
v3 = (gamma3**2)*(t**2)*np.exp(-gamma3*t)

omega=np.linspace(0,10,100)
gamma=1
t=np.linspace(-10,10,1000)
for n in omega:
    if n<0:
        v=0
    if n>0 and n<(2/t):
        v=((2*omega)/gamma)*(np.exp(-gamma*t)-2*(np.exp(gamma*t)/2)+1)
    if n>2/omega:
        v=((2*omega)/gamma)*np.exp(-gamma*t)*(np.exp((2*gamma)/omega)-1)**2

plt.plot(omega,v)
plt.show()
'''



