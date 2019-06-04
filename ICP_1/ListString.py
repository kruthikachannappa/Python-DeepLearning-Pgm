from pip._vendor.distlib.compat import raw_input


#takes the input and coverts to string
input_string = raw_input ("Enter number of elements : ")
str_list = list(input_string)
#removes the unwanted elements and updated the list
str_list = [element for element in str_list if element not in (str_list[1],str_list[2])]
#reverses the list and joins it
out_str = ''.join(reversed(str_list))
print(out_str)

