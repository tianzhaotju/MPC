for j in [10, 20, 30, 40, 50, 60]:
    for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        f = open("./log/test_future_Net_s" + str(i)+'_f'+str(j) + ".txt")
        lines = f.readlines()
        result = []
        for k in [-5,-4,-3,-2,-1]:
            a = lines[k].split(":")
            print(a)
        exit()
