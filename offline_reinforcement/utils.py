from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import time
import json


class StatusPrinter:

    def __init__(self):

        self.elements = {}

    def add_bar(self, name: str, statement: str, max_value: int, num_blocks: int=30, value: int=0, bold: bool=False):

        self.elements[name] = ("\033[1;4m"*bold+statement+"\033[0m", ProgressBar(max_value, num_blocks, value), "bar")

    def add_counter(self, name: str, statement: str, max_value: int, value: int=0, bold: bool=False):

        self.elements[name] = ("\033[1m"*bold+statement+"\033[0m", ProgressBar(max_value, 1, value), "counter")

    def increment_and_print(self, name):

        if self.elements[name][2] == "counter": 
            self.elements[name][1].increment()
            print(self.elements[name][0]+": {}/{}".format(self.elements[name][1].value, self.elements[name][1].max_value))

        elif self.elements[name][2] == "bar": 
            prev_b = self.elements[name][1].blocks
            self.elements[name][1].increment()
            curr_b = self.elements[name][1].blocks
            if (self.elements[name][1].value == 1 or
                prev_b != curr_b or
                self.elements[name][1].value == self.elements[name][1].max_value):
                print("  "+str(self.elements[name][1]), end="\r" if self.elements[name][1].value < self.elements[name][1].max_value  else "\n")

    def print_statement(self, name):
        
        print(self.elements[name][0]+":")

    def reset_element(self, name):
        
        self.elements[name][1].reset()


class ProgressBar:

    def __init__(self, max_value: int, num_blocks: int, value: int=0):

        self.max_value = max_value
        self.num_blocks = num_blocks
        self.block_size = max_value / num_blocks
        self.value = value 
        self.blocks = int(self.value / self.block_size)

    def increment(self):
        self.value += 1
        self.blocks = int(self.value / self.block_size)

    def reset(self):
        self.value = 0
        self.blocks = 0

    def __str__(self):
        return "|"+"█"*self.blocks+" "*(self.num_blocks - self.blocks)+"|"


class Parameters:

    def __init__(self, path=None):
        
        self.fixed = False
        self.help = {}

        if path:
            self.load_from_file(path)

    def fix(self):
        
        self.fixed = True

    def load_from_file(self, path):
        
        with open(path) as f:
            data = json.load(f)
        for key in data.keys():
            setattr(self, key, data[key][0])
            self.help[key] = data[key][1]

    def __setattr__(self, name, value):

        if name != 'fixed' and self.fixed and name in self.__dict__:
            raise TypeError("Parameters are already fixed. Not allowed to change them.")

        else:
            self.__dict__[name] = value



def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
