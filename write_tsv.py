#encoding=utf-8
import jieba
import fasttext
'''
f = open("data/train.tsv", 'r', encoding='utf8')
outf = open("data/lab3fenci.csv",'w')

for line in f:
    line=line.strip()
    l_ar=line.split("\t")
    name=l_ar[0]
    ty_pe=l_ar[1].split("--")[2]
    seg_name=jieba.cut(name.replace("\t","").replace("\n",""))
    outline = " ".join(seg_name)
    outline = "__label__" + ty_pe+"\t"+ outline+"\n"
    outf.write(outline)
f.close()
outf.close()
print ("Word segmentation complete.")
'''

classifier=fasttext.load_model('lab3fenci.model.bin',label_prefix='__label__')

ff = open("data/test2.tsv",'w') #打开一个文件可写模式
with open("data/test1.tsv",'r') as f:  #打开一个文件只读模式
    next(f)
    line = f.readlines()
    for line_list in line:
        line_new =line_list.replace('\n','') #将换行符替换为空('')
        seg_name = jieba.cut(line_new) #jieba分词处理
        outline = classifier.predict([" ".join(seg_name)]) #构成字符串 传入分类器进行预测
        line_new=line_new+r'|'+'\t'+outline[0][0]+'\n'  #添加"|"和预测得到的标签,同时加上"\n"换行符
        #print(line_new) #控制台显示
        ff.write(line_new) #写入一个新文件中
#classifier=fasttext.supervised('data/lab3fenci.csv','lab3fenci.model',epoch=100,dim=200,bucket=500000)#分词结果传入监督学习模型，生成模型文件  训练100次
#print (classifier.labels)
#classifier=fasttext.load_model('lab3fenci.model.bin',label_prefix='__label__')
#print (classifier.labels)
#print(classifier.loss_name)
#result = classifier.test('data/lab3fenci.csv')
#print (result.precision)
#print (result.recall)

'''
texts = [" ".join(jieba.cut('诗瑞小型犬钙奶味狗粮天然粮泰迪贵宾哈士奇金毛犬主粮大型犬成犬粮牛肉蔬菜全犬离乳期奶糕 钙奶味大型犬幼犬粮10公斤'))]
print(texts)
labels = classifier.predict(texts)    
print (labels)
'''

