from colour import Color

red = Color("#E74C3C")
green = Color("#2ECC71")
yellow = Color("#F1C40F")

def color_card(num):
    colors = [ x.hex_l for x in list(red.range_to(yellow,50)) + list(yellow.range_to(green,50))]
    return colors[int(num)]

if __name__ == '__main__':
    print(color_card(2))