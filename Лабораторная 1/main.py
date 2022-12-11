from typing import Union, Any
import numpy as np
import doctest


class SimpleNeuralNetwork:
    def __init__(self, seed: int):
        '''
        Defines random seed for examples representation
        :param seed: number as seed

        Examples:
        >>> nn = SimpleNeuralNetwork(50)
        '''
        if not isinstance(seed, int):
            raise TypeError(f'seed must be an integer, got {type(seed)}')
        self._seed = seed
        np.random.seed(self._seed)

        # initiate weights
        self.weights = 2 * np.random.random((3, 1)) - 1

    def _sigmoid(self, x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
        '''
        Implements sigmoid activation on the layer output
        :param x: operating variable

        Examples:
        >>> nn = SimpleNeuralNetwork(1)
        >>> answer = nn._sigmoid(np.array([[1, 3, 4, 5]]))
        '''
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
        '''
        Computes sigmoid derivative
        :param x: operating variable

        Examples:
        >>> nn = SimpleNeuralNetwork(1)
        >>> answer = nn._sigmoid(np.array([[1, 3, 4, 5]]))
        >>> answer_der = nn._sigmoid_derivative(answer)
        '''
        if not isinstance(x, (int, float, np.ndarray)):
            raise TypeError(f'x must be an integer or float, got {type(x)}')
        return x * (1 - x)

    def _train(
            self,
            inputs: np.array,
            outputs: np.array,
            iterations: int) -> np.array:
        '''
        Trains model to get best result on train set
        :param inputs: input values
        :param outputs: expected train values
        :param iterations: number of steps

        Examples:
        >>> nn = SimpleNeuralNetwork(10)
        >>> inputs = np.array([[1, 0, 1],[0, 1, 1],[1, 1, 1],[1, 1, 0]])
        >>> outputs = np.array([[1, 1, 0, 0]]).T
        >>> nn._train(inputs, outputs, 100)
        '''
        if not isinstance(iterations, int):
            raise TypeError(f'iterations must be an integer, got {type(iterations)}')
        for _ in range(iterations):
            if not isinstance(inputs, np.ndarray):
                raise TypeError(f'inputs must be an np.array, got {type(inputs)}')
            output = self._process(inputs)
            if not isinstance(outputs, np.ndarray):
                raise TypeError(f'outputs must be an np.array, got {type(inputs)}')
            error = outputs - output

            adjust = np.dot(inputs.T, error * self._sigmoid_derivative(output))
            self.weights += adjust

    def _process(self, inputs: Union[int, float, np.ndarray]):
        '''
        calculates output from the layer after activation
        :param inputs: input values

        Examples:
        >>> nn = SimpleNeuralNetwork(3)
        >>> inputs = np.array([[1, 0, 1],[0, 1, 1],[1, 1, 1],[1, 1, 0]])
        >>> outputs = np.array([[1, 1, 0, 0]]).T
        >>> nn._train(inputs, outputs, 1000)
        >>> test_inputs = np.array([1, 0, 0])
        >>> answer = nn._process(test_inputs)
        '''
        if not isinstance(inputs, (int, float, np.ndarray)):
            raise TypeError(f'inputs must be an np.array, got {type(inputs)}')
        inputs = inputs.astype(float)
        output = self._sigmoid(np.dot(inputs, self.weights))

        return output


class Queue:
    def __init__(self):
        '''
        initiates queue datatype

        Examples:
        >>> test = Queue()
        '''
        self.items = []

    def is_empty(self):
        '''
        Check if queue object is empty
        :return: True if queue is empty, else false

        Examples:
        >>> test = Queue()
        >>> res = test.is_empty()
        '''
        return self.items == []

    def enqueue(self, item: Any) -> None:
        '''
        Adds element 'item' to the rear of the queue
        :param item: any element
        :return: None

        Examples:
        >>> test = Queue()
        >>> test.enqueue('5')
        '''
        self.items.insert(0, item)

    def dequeue(self):
        '''
        Drop the first element from the queue
        :return: None

        Examples:
        >>> test = Queue()
        >>> test.enqueue(5)
        >>> test.enqueue(3)
        >>> five = test.dequeue() # drop 5
        '''
        return self.items.pop()

    def __len__(self):
        '''
        Returns the len of the queue
        :return: Length of the queue

        Examples:
        >>> test = Queue()
        >>> test.enqueue(1)
        >>> length = len(test)
        '''
        return len(self.items)


class Stack:
    def __init__(self):
        '''
        Initialize __index that will hold items in the stack
        :return: None

        Examples:
        >>> stack = Stack()
        '''
        self.__index = []

    def __len__(self) -> int:
        '''
        Calculate the length of the Stack
        :return: length of the stack

        Examples:
        >>> test = Stack()
        >>> length = len(test)
        '''
        return len(self.__index)

    def push(self, item: Any) -> None:
        '''
        Adds item to the end of the stack
        :param item: any element
        :return: None

        Examples:
        >>> test = Stack()
        >>> test.push(5)
        '''
        self.__index.insert(0, item)

    def peek(self):
        '''
        Returns top element of the stack
        :return: top element of the stack

        Examples:
        >>> test = Stack()
        >>> test.push(1)
        >>> test.push(2)
        >>> top = test.peek() # 2
        '''
        if len(self) == 0:
            raise Exception('Execute elements from empty stack')
        return self.__index[0]

    def pop(self):
        '''
        Drops out top element from the stack
        :return: dropped top element

        Examples:
        >>> test = Stack()
        >>> test.push(1)
        >>> test.push(2)
        >>> top = test.pop() # 2
        '''
        if len(self) == 0:
            raise Exception('Execute elements from empty stack')
        return self.__index.pop(0)

    def __str__(self):
        '''
        Returns adequate description what Stack() class contains
        :return: description of the object

        Examples:
        >>> test = Stack()
        >>> test.push('1')
        >>> description = str(test) # "['1']"
        '''
        return str(self.__index)


if __name__ == "__main__":
    doctest.testmod()
