
name = 'time15.txt'
f = open(name, 'r')

total = 0
count = 0
for line in f.readlines():
    line2 = line.split('Time ')
    if len(line2) == 2:
        time_str = line2[1].split(' ')[0]
        time = float(time_str)
        total += time
        print(time)
        count += 1

print(total)
print(count)