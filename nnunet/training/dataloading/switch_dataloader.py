class switchable_generator(object):
    def __init__(self,gens,default_id=0):
        self.gens = gens
        self.cur_gen = gens[default_id]

    def setPart(self,id):
        self.cur_gen = self.gens[id]

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.cur_gen)
    
    def next(self):
        return self.__next__()