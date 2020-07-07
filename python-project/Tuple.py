from collections import defaultdict
import operator

def tuple():
    list1 = [( 'John', ('Physics', 80)) , ('Daniel', ('Science', 90)),
    ('John', ('Science', 95)), ('Mark',('Maths', 100)),
    ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]
    # intialize the dict 
    dict1 = defaultdict(list)
    # Iterate through the values and assign it to the same key
    for key , val in list1:
        dict1[key].append(val)
    # Sort the dictionary on basis of key
    sorted_dict = sorted(dict1.items(), key=operator.itemgetter(0))
    print(sorted_dict)



if __name__ == '__main__':
    tuple()