class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def describe(self):
        print(f'{self.name} is a dog')


class Pet(Animal):
    def __init__(self, name, species, owner):
        super().__init__(name, species)
        self.owner = owner

    def greet(self):
        print(f'{self.name} belongs to {self.owner}')


class Dog(Pet):
    def __init__(self, name, species, owner, breed):
        super().__init__(name, species, owner)
        self.breed=breed
    
    def bark(self):
        print(f'{self.name} the {self.breed} says: Woof')

dog1=Animal('rex', 'Canis lupus familiaris')
dog1=Pet('rex', 'Canis lupus familiaris','brad')
dog1=Dog('rex', 'Canis lupus familiaris','brad','French Bulldog')

dog1.describe()
dog1.greet()
dog1.bark()
