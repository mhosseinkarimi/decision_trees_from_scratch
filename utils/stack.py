class Stack:
    def __init__(self,):
        self.stack = []
    
    def push(self, item):
        self.stack.append(item)
    
    def pop(self):
        if self.size() > 0:
            return self.stack.pop()
    
    def size(self):
        return len(self.stack)
    
    def empty(self):
        self.stack = []
    
    def top(self):
        if self.size() > 0:
            return self.stack[-1]
    
    def __repr__(self) -> str:
        return self.stack.__repr__()
    

