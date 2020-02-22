# -*- coding: utf-8 -*-
"""
爬虫 大专家
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import os
import time
import random
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--htmlPath", default="DZJ_html_txt",
                  help="the path to save the parsed html", metavar="FILE")
parser.add_option("--outPath", default="DZJ_outfile",
                  help="the output file path ")
parser.add_option("--chromedriver", default='chromedriver.exe',
                  help='the path of chromedriver')
(options, args) = parser.parse_args()

html_txt_save_path = options.htmlPath
if not os.path.exists(html_txt_save_path):
    os.mkdir(html_txt_save_path)


outfile_path = options.outPath
if not os.path.exists(outfile_path):
    os.mkdir(outfile_path)


chromedriver_loc = options.chromedriver
def get_url(url='https://www.dazhuanjia.com/edu/market/case/view/1252004', chromedriver_loc=chromedriver_loc):
    ''' 输入 url: eg.  https://www.dazhuanjia.com/edu/market/case/view/1252004
        输出：保存对应html
    '''

    assert os.path.isfile(chromedriver_loc) is True, 'please download chromedriver and move it to corresponding file'
    options = webdriver.ChromeOptions()  # 为了不弹出窗口
    options.add_argument("headless")
    browser = webdriver.Chrome(executable_path=chromedriver_loc, chrome_options=options)     # 打开 Chrome 浏览器
    browser.get(url)  # 打开url网页
    
    time.sleep(2+random.randint(0,10)*0.1)  # 貌似直接里面得到page_source会出问题,需要等几秒
    html = browser.page_source       # get html [这样保存的html会有文字]
    
    filename = url.split('/')[-1]
    # 解析这个html
    parse_content(html, filename)
    
    # 把这个html保存
    with open(os.path.join(html_txt_save_path, filename+".txt"), "w") as f:
        f.write(html)


def parse_content(html, filename):
    # ''' 基于get_url得到的有内容的html，解析 ''''
    soup = BeautifulSoup(html, features='lxml')
    
    # 精选摘要
    try:
        abstract = soup.find_all('p', {'class': 'row-content'})[1].string.strip()
    except:
        print('no abstract')
        abstract = ''
    
    # 疾病信息
    try:
        info = soup.find_all('div', {'class': 'disease-info-detail-col'})   #（典型症状+主诉+现病史+现病时长）
        if info == []:
            info = soup.find_all('div',{'class':'disease-info-item'})
            disease_string = ''   # 只把现病史提取出来
            for w in info[0].strings:
                disease_string += w + ' '
        else:
            disease_string = ''   # 只把现病史提取出来
            for w in info[2].strings:
                disease_string += w + ' '
    except:
        print('no disease string')
        disease_string = ''

    # 辅助检查
    try:
        check = soup.find_all('div',{'class':'disease-info-item'}) 
        check_string = ''
        for w in check[2].strings:
            check_string += w
    except:
        print('no check')
        check_string = ''
    
    # 把以上的内容分别写出
    with open(os.path.join(outfile_path, filename+'_abstract.txt'), 'w', encoding='utf-8') as f1:
        f1.write(abstract)
    
    with open(os.path.join(outfile_path, filename+'_disease.txt'), 'w', encoding='utf-8') as f2:
        f2.write(disease_string)
    
    with open(os.path.join(outfile_path, filename+'_check.txt'), 'w', encoding='utf-8') as f3:
        f3.write(check_string)
    

if __name__ == '__main__':
    start_id = 1000100
    for i in range(start_id, start_id+int(3e5)):
        print('============{}================='.format(i))
        url = 'https://www.dazhuanjia.com/edu/market/case/view/{}'.format(i)
        try:
            get_url(url)
        except:
            print('sth wrong with {}'.format(url))
