import statistics
a = [[1,2,3],[0,0,0]]
b = []
for i in range(len(a)):
    b = b + a[i] 
c = statistics.median(b)

print(c)