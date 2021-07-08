def split(dataset, perc):

    flag = int(len(dataset) * perc / 100)

    x_train = dataset[0:flag]

    x_test = dataset[flag:len(dataset)]

    return x_train, x_test
