
def convert_yaml2grid(yaml):
    grid = [tuple([tuple(yaml[k1][k2]) for k2 in list(yaml[k1])]) for k1 in list(yaml)]

    return grid


def return_empty_list():
    return []
