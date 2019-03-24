#encoding=utf-8
import jieba
import fasttext

f = open("data/newtrain.tsv", 'r', encoding='utf8')
outf = open("data/newlab3fenci.csv",'w')

for line in f:
    line=line.strip() #去除空格
    l_ar=line.split("\t") #对读入的行 由缩进 分割两部分
    name=l_ar[0] #前一部分
    ty_pe=l_ar[1] #第二部分
    seg_name=jieba.cut(name.replace("\t","").replace("\n","")) #去除name中的缩进和换行后进行jieba分词
    outline = " ".join(seg_name) #列表组成字符串
    outline = "__label__" + ty_pe+"\t"+ outline+"\n" #添加分类标识符 重组
    outf.write(outline) #写入文件
f.close()
outf.close()
print ("Word segmentation complete.")

classifier=fasttext.supervised('data/newlab3fenci.csv','nwelab3fenci.model',epoch=50,dim=200,bucket=500000) #分词结果传入监督学习模型，生成模型文件  训练100次
result = classifier.test('data/newlab3fenci.csv')
print (result.precision) #准确率
print (result.recall) #召回率
