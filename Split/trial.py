
from Storing import store

a = [1,[2],[3],4,5]
store("first.dat", a)

from Reading import read

b = read("first.dat")
print(b)