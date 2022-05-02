class classproperty:
    """Class property"""

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


# The following replacement works only for python 3.9
# def classproperty(func):
#     return classmethod(property(func))
