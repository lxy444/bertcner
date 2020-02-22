# -*- coding: utf-8 -*-
"""
快速问医生 提问部分 https://www.120ask.com/question
"""

from bs4 import BeautifulSoup
from optparse import OptionParser
import urllib


parser = OptionParser()
parser.add_option("--failLog", default="AskQue_fail_log.txt",
                  help="the file to record the web which can not be parsed", metavar="FILE")
parser.add_option("--outPath", default="AskQue_out.txt",
                  help="the output file name ")
parser.add_option('--minLen', default=5,
                  help="the line whose character num is less than minLen is removed here")
parser.add_option('--maxPage', default=200,   
                  help="the upper bound of pages for department_url")
(options, args) = parser.parse_args()

headers = ('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.86 Safari/537.36')
opener = urllib.request.build_opener()
opener.addheaders=[headers]

fail_log = open(options.failLog, 'w', encoding='utf-8')  
out_file = open(options.outPath, 'w', encoding='utf-8')
min_len = options.minLen
max_pages = options.maxPage


def process_department_url(department_url):
    """
    Get the question urls 
    
    Args:
        department_url: such url as https://www.120ask.com/list/neike/
    Returns:
        the question urls of the department  
    """
    html = opener.open(department_url).read().decode('utf-8') 
    soup = BeautifulSoup(html, features='lxml')
    
    # 判断一下是否有最后一页
    pages = soup.find('div',{'class':'clears h-page'})
    contain_last_page = False
    last_page_id = None
    for page in pages.find_all('a'):
        page_str = page.string
        if page_str=='最后一页':
            contain_last_page = True
            last_page_id = int(page['href'].split('/')[-2])
    
    if not contain_last_page:   #不包含“最后一页”就不处理这个科室对应的内容了
        return None
    
    # 下面得到这个科室下所有问题对应的url
    que_url_lst = []
    
    for page_id in range(1, min(last_page_id+1, max_pages+1)):
        one_page_url = department_url+str(page_id)
        one_page_html = opener.open(one_page_url).read().decode('utf-8') 
        one_page_soup = BeautifulSoup(one_page_html, features='lxml')
    
        for que in one_page_soup.find_all('a',{'class':"q-quename"}):
            que_url = 'https:'+ que['href']
            que_url_lst.append(que_url)
    return que_url_lst
    

def process_question_url(question_url):
    """
    Process the question_url and only get the reply.
    
    Args:
        question_url: the url
    """
    html = opener.open(question_url).read().decode('utf-8') 
    soup = BeautifulSoup(html, features='lxml')
    ans_lst = soup.find_all('div',{'class':'crazy_new'})
    
    out_string = ''   #这个问题下的所有回答拼接在一起（不同回答换行连接）
    for ans in ans_lst:
        ans = ans.find('p')
        
        ans_str = ' '.join(list(filter(lambda s: len(s)>min_len,[item.strip() for item in list(ans.strings)])))
        out_string += ans_str +'\n'
    return out_string
        

def main(start_url='https://www.120ask.com/question'):
    """
    Args:
        start_url: the web url
    """
    
    # 找到内科、外科等对应链接
    html = opener.open(start_url).read().decode('utf-8')  # if has Chinese, apply decode()
    soup = BeautifulSoup(html, features='lxml')
    
    lst = soup.find_all('div',{'class':'clearfix'})  # 像内科还具体分为呼吸内科、消化内科等，得到这些具体链接
    url_lst = [] #保存诸如 呼吸内科等链接
    for department in lst[2:]:  
        small_departments = [item.find('a')['href'] for item in department.find_all('li')]  
        if small_departments!=[]:
            url_lst += small_departments
        else:
            url_lst.append(department.find('a')['href'])
  
    # 对每个科室，得到这个科室对应的所有提问链接
    for department_url in url_lst:
        department_url = 'https:'+department_url
        print(f'Department {department_url} url')
        try:
            que_url_lst = process_department_url(department_url)  
        except:
            print(f'Department {department_url} does not contain such click as [last page]')
            continue
        
        if que_url_lst==None:
            print(f'Department {department_url} does not contain such click as [last page]')
            continue
        else:
            for que_url in que_url_lst:
                try:
                    ans_out = process_question_url(que_url)
                    out_file.write(ans_out)
                except:
                    fail_log.write(que_url+'\n')
                
    fail_log.close()
    out_file.close()
        
    
if __name__=='__main__':
    main()