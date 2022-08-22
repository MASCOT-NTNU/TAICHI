class A:
    def __init__(self):
        self.__a = 1


class B(A):
    def __init__(self):

        super().__init__()
        print(self.__a)
        self._a = 2
        print(self._a)
        pass

b = B()
# print(a.__a)

