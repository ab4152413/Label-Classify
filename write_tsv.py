#encoding=utf-8
import jieba
import fasttext

classifier=fasttext.load_model('nwelab3fenci.model.bin',label_prefix='__label__')

ff = open("data/test2.tsv",'w') #打开一个文件可写模式
with open("data/test.tsv",'r') as f:  #打开一个文件只读模式
    next(f)
    line = f.readlines()
    for line_list in line:
        line_new =line_list.replace('\n','') #将换行符替换为空('')
        seg_name = jieba.cut(line_new) #jieba分词处理
        outline = classifier.predict([" ".join(seg_name)]) #构成字符串 传入分类器进行预测
        line_new=line_new+'\t'+outline[0][0]+'\n'  #添加预测得到的标签,同时加上"\n"换行符
        print(line_new) #控制台显示
        ff.write(line_new) #写入一个新文件中
