import random

# Change brightness of image
def brightChange(data, val):
    if (val == 0):
        val = random.randrange(-50, 50)

    x = 0
    y = 0

    while (y < len(data)):
        while (x < len(data[y])):
            data[y][x] = (data[y][x] + val) % 255
            for i in range(3):
                if data[y][x][i] < 0:
                    data[y][x][i] = 0
            x += 1
        y += 1

    return data