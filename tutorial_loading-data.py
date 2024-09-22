print('Hello World')
a=10
b='hi there'
c='hey there!'
print(b,a,c)
print(b+c)
#name=input("Who are you? ")
#print(c,name)
#age=int(input("How old are you? "))
#print("Right, so you are",name+", age:",age,"\byrs")
for i in range(1,101):
	print(i,end=" ")
	if(i%10==0):
		print()
# ividenn aan shab code--------------------------------------
def wordify(number):
	multipliers={1:'',100:'hundred',1000:'thousand',1000000:'million',1000000000:'billion',1000000000000:'trillion'}
	sums={0:"",1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten',11:'eleven',12:'twelve',13:'thirteen',14:'fourteen',15:'fifteen',16:'sixteen',17:'seventeen',18:'eighteen',19:'nineteen',20:'twenty',30:'thirty',40:'fourty',50:'fifty',60:'sixty',70:'seventy',80:'eighty',90:'ninety'}
	mult=1000000000000
	quotient=number//mult
	if quotient>0:
		wordify(quotient)		#Incase the number is more than trillions
		print(multipliers[mult])	#print trillion
	while number>0:
		mult/=1000
		quotient=number//mult		#isolating the units, first billions then millions.....
		number%=mult			#removing the unit we isolated, from the number
		if quotient==0:
			continue
		if quotient//100!=0:		#if there's a hundred's digit
			print(sums[quotient//100],end=" ")
			print("Hundred",end=" ")
		if quotient%100<=20:		#for 'one' to 'twenty'
			print(sums[quotient%100],end=" ")
		else :				#for 'twenty one' to 'ninety nine'
			print(sums[(quotient%100)-(quotient%10)],end=" ")
			print(sums[quotient%10],end=" ")
		
		print(multipliers[int(mult)],end=" ") #to print billion, million, thousand

number=int(input("\n\nEnter the number: "))
if number==0:
	print("Number in words: zero")
else :
	print("Number in words is:",end=" ")
	wordify(number)
print("")
#--------------------------------------------------------------
