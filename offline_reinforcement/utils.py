from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import time
import json


class StatusPrinter:

    def __init__(self):

        self.elements = {}

    def add_bar(self, name: str, statement: str, max_value: int,
                num_blocks: int = 30, value: int = 0, bold: bool = False):

        self.elements[name] = ("\033[1;4m"*bold+statement+"\033[0m", ProgressBar(max_value, num_blocks, value), "bar")

    def add_counter(self, name: str, statement: str, max_value: int, value: int = 0, bold: bool = False):

        self.elements[name] = ("\033[1m"*bold+statement+"\033[0m", ProgressBar(max_value, 1, value), "counter")

    def increment_and_print(self, name: str):

        if self.elements[name][2] == "counter":
            self.elements[name][1].increment()
            print(self.elements[name][0]+": {}/{}".format(self.elements[name][1].value,
                                                          self.elements[name][1].max_value))

        elif self.elements[name][2] == "bar":
            prev_b = self.elements[name][1].blocks
            self.elements[name][1].increment()
            curr_b = self.elements[name][1].blocks

            if (self.elements[name][1].value == 1 or
                prev_b != curr_b or
                self.elements[name][1].value == self.elements[name][1].max_value):

                    print("  "+str(self.elements[name][1]), end="\r" if
                                   self.elements[name][1].value < self.elements[name][1].max_value else "\n")

    def print_statement(self, name: str):

        print(self.elements[name][0]+":")

    def reset_element(self, name: str):

        self.elements[name][1].reset()


class ProgressBar:

    def __init__(self, max_value: int, num_blocks: int, value: int = 0):

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

    def __init__(self, path: str = None, is_help: bool = False):

        self.fixed = False

        if not is_help:
            self.help = Parameters(is_help=True)

        if path:
            self.load_from_file(path)

    def fix(self):

        self.fixed = True
        self.help.fixed = True

    def load_from_file(self, path: str):

        with open(path) as f:
            data = json.load(f)
        for key in data.keys():
            setattr(self, key, data[key][0])
            setattr(self.help, key, data[key][1])

    def __setattr__(self, name: str, value):

        if name != 'fixed' and self.fixed and name in self.__dict__:
            raise TypeError("Parameters are already fixed. Not allowed to change them.")

        else:
            self.__dict__[name] = value

    def __str__(self):

        out = "<table> <thead> <tr> <th> Parameter </th> <th> Value </th> </tr> </thead> <tbody>"
        for name in self.__dict__.keys():
            if name != 'help' and name != 'fixed':
                out += " <tr> <td> "+name+" </td> <td> {} </td> </tr> ".format(getattr(self, name))
        out += "</tbody> </table>"
        return out
    
    def __repr(self):

        return str(self)









