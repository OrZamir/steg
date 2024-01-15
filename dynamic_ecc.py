
#################################################
# This class implements the Dynamic ECC of      #
# Section 5 in the paper                        #
#################################################

class DynamicECC:
    def __init__(self, input):
        self.input = input
        self.last_index_written = -1
        self.suffix_to_remove = 0
        self.stream = []
        self.default_symbol = "0"

    @staticmethod
    def decode(stream):
        message = []
        for symbol in stream:
            if symbol == "<":
                if message:
                    message.pop()
            else:
                message.append(symbol)

        return "".join(message)

    def update(self, symbol):
        last_index_written = self.last_index_written
        suffix_to_remove = self.suffix_to_remove

        self.stream.append(symbol)

        # There is an incorrect suffix
        if suffix_to_remove:
            if symbol == "<":
                self.suffix_to_remove = suffix_to_remove - 1
            else:
                self.suffix_to_remove = suffix_to_remove + 1
            return None

        # The previous stream is correct
        next_symbol = self.input[last_index_written+1] if (last_index_written + 1 < len(self.input)) \
            else self.default_symbol

        if symbol == next_symbol:
            self.last_index_written = last_index_written + 1
        elif symbol == "<":
            if last_index_written > -1:
                self.last_index_written = last_index_written - 1
        else:
            self.suffix_to_remove = 1
        return None

    def next_symbol(self):
        if self.suffix_to_remove:
            return "<"
        else:
            next_symbol = self.input[self.last_index_written + 1] if (self.last_index_written + 1 < len(self.input)) \
                else self.default_symbol
            return next_symbol
