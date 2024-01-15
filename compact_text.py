
#####################################################
# This class implements a simple encoding and       #
# decoding of English text and some special         #
# characters, that uses 5 bits per letter           #
#####################################################

class CompactText:
    ENCODING = (' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                'X', 'Y', 'Z', '@', ':', '.', '/', '?')

    @staticmethod
    def text_to_bits(text):
        enc_map = {CompactText.ENCODING[i]: i for i in range(32)}
        bits = ""
        for char in text:
            bits += ("0"*5 + bin(enc_map[char])[2:])[-5:]
        return bits

    @staticmethod
    def bits_to_text(bits):
        dec_map = {i: CompactText.ENCODING[i] for i in range(32)}
        text = ""
        for ind in range(0, len(bits), 5):
            text += dec_map[int(bits[ind : ind+5], 2)]
        return text
