import math
import random
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def inverseColor(r, g, b):
    h, s, v = rgb2hsv(r,g,b)
    # h = ((random.random()+0.618)%1)*360
    h = (h + 180) % 360
    # h = 360 - h
    s = 0.618
    v = 0.95
    return hsv2rgb(h, s, v)

def randColor():
    # h, s, v = rgb2hsv(r,g,b)
    h = ((random.random() + 0.618) % 1) * 360
    # h = (h + 180)%360
    # h = 360 - h
    s = 0.618
    v = 0.95
    return hsv2rgb(h, s, v)

def contrastColor(r, g, b):
    h,s,v = rgb2hsv(r,g,b)
    h1 = (h + 120) % 360
    h2 = (h + 240) % 360
    s = 0.618
    v = 0.95
    r1, g1, b1 = hsv2rgb(h1, s, v)
    r2, g2, b2 = hsv2rgb(h2, s, v)
    return r1, g1, b1, r2, g2, b2