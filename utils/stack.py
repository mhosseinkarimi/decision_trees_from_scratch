from typing import Any
class Stack:
    """A simple implementation of stack. Stack is LIFO (Last In First Out), data structure.
    """

    def __init__(self,):
        """Stack in this project is implemeted using list.
        """
        self.stack = []

    def push(self, item: Any) -> None:
        """Pushing a new member on top of stack.

        Args:
            item (Any): The item that is pushed to the stack
        """
        self.stack.append(item)

    def pop(self) -> Any:
        """pops the top element in the stack

        Returns:
            Any: The top element of the stack is removed from stack and returned
        """
        if self.size() > 0:
            return self.stack.pop()

    def size(self):
        return len(self.stack)

    def empty(self):
        """Empties the stack.
        """
        self.stack = []

    def top(self) -> Any:
        """
        Returns:
            Any: The top element of the stack without removing it
        """
        if self.size() > 0:
            return self.stack[-1]

    def __repr__(self) -> str:
        return self.stack.__repr__()
