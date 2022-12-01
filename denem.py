def cutTheSticks(arr):
    sticks_cut = []

    while max(arr)> 0:
        cutted_sticks = 0
        min = max(arr)
        for e in arr:
            if e > 0 and e<min :
                min = e
        for i in range(len(arr)):
            if arr[i] > 0:
                cutted_sticks += 1
            arr[i] -= min
        sticks_cut.append(cutted_sticks)

    return sticks_cut

res = cutTheSticks([5,4,4,2,2,8])
print(res)