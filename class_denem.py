"""
class Person:
    def __init__(self,name,age,size,workday):
        self.name = name
        self.age = age
        self.size = size
        self.pension = workday
        print("init başlatıldı.")

    def intro(self,name,age,size,workday):
        res = (name,age,size,workday)
        print(f'name: {name}\nage: {age}\nsize: {size}\nworkday: {workday}')

    def CalculateBornYear(self,age):
        res = 2024 - age
        print(f'Your born year is {res}')
    
    def CalculateSize(self,size):
        if size <=160:
            print("You are short.")
        elif size>160 and size<=180:
            print("You are normal person.")
        elif size > 180:
            print("You are tall.")
        else: 
            print("You entered wrong value for your size.")
    
    def CalculateRetired(self,workday):
        retired = 5000-workday
        year = int(retired/365)
        day = retired%365
        print(f"You need work {year} year and {day} day.")

nam = input('Please enter your name: ')
ag = int(input('Please enter your age: '))
siz = int(input('Please enter your size: '))
work = int(input('Please enter your work day: '))

who = Person(nam,ag,siz,work)

#who.intro(nam,ag,siz,work)
#who.CalculateBornYear(ag)
who.CalculateRetired(work)
#who.CalculateSize(siz)
"""
"""
class top:
    def __init__(self):
        pass
    
    def toplama(self,x, y):
        z = x + y
        return z

    def sayi(self):
        x = 5
        y = 10
        sonuc = self.toplama(x, y)
        print(sonuc)

nesne=top()
nesne.sayi()
"""

