from pip._vendor.distlib.compat import raw_input

#takes the input and coverts to string
input_string = raw_input ("Enter the sentence : ")

#splits the words of sentence and put into list
str_list = list(input_string.split())

#iterated over the list and checked for "python" and replaced
for i, item in enumerate(str_list):
        if "python" in item:
            str_list[i] = "pythons"

#joined the words back with spaces in between each word
output_string = ' '.join(str_list)

print(output_string)