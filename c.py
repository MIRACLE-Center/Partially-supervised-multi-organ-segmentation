num = input()
num = int(num)
point = {}
for i in range(num):
    xy_str = input()
    xy = xy_str.split(' ')
    point[int(xy[0])]=int(xy[1])
k = point.keys()
k = list(k)
k.sort(reverse=True)
y_max = -1
result = []
for i in k:
    if point[i]>y_max:
        result.append(i)
        y_max = point[i]
result.sort()
for i in result:
    print(f"{i} {point[i]}")