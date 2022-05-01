class classproperty:
    """Class property"""

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


# python3.8 doesn't like the definition of class property below
# with python3.9 runs fine ...

# def classproperty(func):
#     return classmethod(property(func))
