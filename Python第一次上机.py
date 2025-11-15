# 01
score1 = float(input("请输入第一门功课的分数: "))

score2 = float(input("请输入第二门功课的分数: "))

score3 = float(input("请输入第三门功课的分数: "))

sum = score1 + score2 + score3

avg = sum / 3

print("三门功课的总分是:", sum)
print("三门功课的平均分是:", avg)



# 02
# 从键盘输入一个三位整数
num = int(input("请输入一个三位整数："))

# 分离出百位、十位和个位数字
# 百位数字
A = num // 100
# 十位数字
B = (num % 100) // 10
# 个位数字
C = num % 10

# 计算反序数
reverse_num = 100 * C + 10 * B + A

# 输出结果
print("这个三位数的反序数为：", reverse_num)
