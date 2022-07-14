# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__


class dotdict(dict):
    """dot.notation access to dictionary attributes""" 
    def __getattr__(*args):         
        val = dict.get(*args) 
        return dotdict(val) if type(val) is dict else val      
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__