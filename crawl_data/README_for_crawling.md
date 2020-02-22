## crawl clinical data for pretraining BERT

- [快速问医生：有问必答(120ask-question part)](https://www.120ask.com/question)

```bash
python crawl_120ask_question.py
```

- [快速问医生：健康经验(120ask-experience part)](http://tag.120ask.com/jingyan)

```bash
python crawl_120ask_experience.py
```

- [壹生/资讯(cmtopdr.com)](https://www.cmtopdr.com/post/)

```bash
python crawl_Report.py
```

- [大专家(dazhuanjia.com)](https://www.dazhuanjia.com/): 

```bash
python crawl_DaZhuanJia.py
```

Noting that the parameters in the code are given. Please modify the location for chromedriver.

- [wenyw.com](http://jibing.wenyw.com/)

```
the code for crawling wenyw.com is avaible in https://github.com/baiyyang/scrapy_medical
```



### crawling radicals

```
python crawl_radical.py
```

the example shows how to get the radical of one Chinese character.
