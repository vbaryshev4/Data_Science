a = {1,2,3,4,5,6,7,8,9}

def build_graph(s):
    result = []
    for i in s:
        for k in s:
            f = (lambda x,y: str(x)+str(y))
            val = int((f(k, i)))
            if val%3 == 0:
                print(
                    '{0} % 3 = 0 with vertices of the graph {1} and {2}'.
                        format(val, k, i)
                    )
                result.append(val)
    return result


if __name__ == '__main__':
    r = build_graph(a)
    r.sort()
    print(r)