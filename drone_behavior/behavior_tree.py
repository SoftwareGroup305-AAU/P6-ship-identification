from abc import ABC, abstractmethod
class Node(ABC):
    @abstractmethod
    def tick(self):
        pass

class ControlNode(Node):
    def __init__(self):
        self.children = []
    
class SequenceNode(ControlNode):
    pass

class FallbackNode(ControlNode):
    pass

class ParallelNode(ControlNode):
    pass

class ActionNode(Node):
    pass

class ConditionNode(Node):
    pass



    
    
        

    

