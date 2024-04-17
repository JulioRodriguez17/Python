num = int(input("Dame un numero: "))
pasos=0

if num > 0:
    while num != 1:
        if num%2 == 0:
            num /= 2
            pasos+=1
        elif num%2 !=0:
            num=3*num+1
            pasos+=1
        print(int(num),end=" ")
    print("pasos=",pasos)
else: 
    print("No se ha podido comprobar la hipotesis")