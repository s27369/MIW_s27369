import numpy as np
import time
import matplotlib.pyplot as plt

# macierz stanów
t1 = ['P','K', 'N']
#macierz wystapien
wyst=np.array([[2,4,1], [0,0,4], [4,1,2]])
#strategia przeciwnika
prob_op = np.array([0.5, 0.1, 0.4])
print(wyst)
print(wyst[0])
print(wyst[0]/sum(wyst[0]))
print(np.random.choice(t1, p=wyst[0]/sum(wyst[0])))

kasa =[]
stankasy=0

n = 30 
state = 'P'

for i in range(n):
    if state == 'P':
        pred = np.random.choice(t1, p=wyst[0]/sum(wyst[0]))
        print(pred)
        print(wyst[0]/sum(wyst[0]))
        stankasy = stankasy +1
        kasa.append(stankasy)
        #odpowiedz na predykcje
        #zobacz rzeczywista akcje przeciwnika
        op_akc = np.random.choice(t1, p=prob_op)
        #zobaczyć wynik gry - pred, op_akc
        #zmodyfikować macierz wystapien
        #przejść do stanu - jaki zagrał przeciwnik: op_akc
        print(op_akc)
    
    if state == 'K':        
        pred = np.random.choice(t1, p=wyst[1]/sum(wyst[1]))
        print(pred)
        op_akc = np.random.choice(t1, p=prob_op)
        print(op_akc)
        stankasy = stankasy +1
        kasa.append(stankasy)
  
    if state == 'N':
        pred = np.random.choice(t1, p=wyst[2]/sum(wyst[2]))
        op_akc = np.random.choice(t1, p=prob_op)
        print(op_akc)
        print(pred)
        stankasy = stankasy +1
        kasa.append(stankasy)
        
    state = op_akc        

    print("\n")
    time.sleep(0.1)

plt.plot(kasa)
plt.show()
print(len(kasa))