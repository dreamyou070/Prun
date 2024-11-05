total_list = [1,2,3]
target = [3]
# substract target list from total list
answer = [x for x in total_list if x not in target]
print(answer)