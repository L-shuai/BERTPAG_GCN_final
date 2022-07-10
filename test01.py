

str = "北京商报记者获悉,美国高端时尚品牌Michael Kors迈克高仕将于<N>月在上海静安嘉里中心开设品牌旗舰店,"
print(str.index("上海静安嘉里中心"))


fin = open("dataset/FinRE/relation2id.txt", 'r', encoding='utf-8', newline='\n', errors='ignore')
lines = fin.readlines()
fin.close()
# id = -1
text = ''
print("len = ",len(lines))
for i in range(0, len(lines)):
    arrs = lines[i].split()
    line = '"'+arrs[0]+'":'+' '+arrs[1]+','
    text+=line

print(text)

# {"unknown":  0, "Create": 1, "Use":  2, "Near":  3,
#                             "Social":  4, "Located":  5, "Ownership":  6, "General-Special":  7, "Family":  8, "Part-Whole":  9}