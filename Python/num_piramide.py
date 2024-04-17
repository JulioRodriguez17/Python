blocks = int(input("Ingresa el número de bloques: "))
height=0
utilizado=1

while height < blocks:
        height+=1
        blocks-=utilizado
        utilizado+=1

print("La altura de la pirámide:", height)
