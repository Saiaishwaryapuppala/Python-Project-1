def Substring(_str):
    temp_str= ""
    dict1= {}
    # Iterate through every letter appending to temporary string and 
    # stops the appending when the next letter in repeated 
    for j in range(len(_str)):
        for i in range(j,len(_str)):
            if not(_str[i] in temp_str):
                temp_str+=_str[i]
            else :
                dict1[temp_str]= len(temp_str)
                temp_str =''
                break
    # After getting all the substrings, search for the max length value
    max_val = max(dict1.values())
    list1=[]
    # Iterate through the list and get the max length substrings without letters repeating
    for key , val in dict1.items():
        if(max_val == val):
            list1.append((key,val))
    print("The longest substrings without repeating characters along with the length")
    print(list1)


if __name__ == '__main__':
    Substring("pwwkew")