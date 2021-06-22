class StatusPrinter:

    def __init__(self):

        self.elements = {}

    def add_bar(self, name: str, statement: str, max_value: int, num_blocks: int=30, value: int=0):

        self.elements[name] = (statement, ProgressBar(max_value, num_blocks, value), "bar")

    def add_counter(self, name: str, statement: str, max_value: int, value: int=0):

        self.elements[name] = (statement, ProgressBar(max_value, 1, value), "counter")

    def increment_and_print(self, name):

        if self.elements[name][2] == "counter": 
            self.elements[name][1].increment()
            print(self.elements[name][0]+": {}/{}".format(self.elements[name][1].value, self.elements[name][1].max_value))

        elif self.elements[name][2] == "bar": 
            bs = int(self.elements[name][1].block_size)
            self.elements[name][1].increment()
            if (self.elements[name][1].value == 1 or
                self.elements[name][1].value % bs == 0 or
                self.elements[name][1].value == 0 or
                self.elements[name][1].value == self.elements[name][1].max_value):
                print("\u001b[?25l"+str(self.elements[name][1]), end="\r" if self.elements[name][1].value < self.elements[name][1].max_value  else "\u001b[?25h\n")

    def print_statement(self, name):
        
        print(self.elements[name][0]+":")

    def reset_element(self, name):
        
        self.elements[name][1].reset()



class ProgressBar:

    def __init__(self, max_value: int, num_blocks: int, value: int=0):

        self.max_value = max_value
        self.num_blocks = num_blocks
        self.value = value 
        self.block_size = max_value / num_blocks

    def increment(self):
        self.value += 1

    def reset(self):
        self.value = 0

    def __str__(self):


        return "|"+"█"*int(self.value/self.block_size)+" "*(self.num_blocks - int(self.value/self.block_size))+"|"


