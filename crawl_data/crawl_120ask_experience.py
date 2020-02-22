# -*- coding: utf-8 -*-
"""
快速问医生 经验部分 https://tag.120ask.com/jingyan/
"""

from bs4 import BeautifulSoup
from optparse import OptionParser
import urllib
import re

parser = OptionParser()
parser.add_option("--failLog", default="AskExp_fail_log.txt",
                  help="the file to record the web which can not be parsed", metavar="FILE")
parser.add_option("--outPath", default="AskExp_out.txt",
                  help="the output file name ")
parser.add_option('--relatedLog', default='AskExp_related_out.txt',
                  help='the output file name of related QA in experience url')
parser.add_option('--minLen', default=5,
                  help="the line whose character num is less than minLen is removed here")
parser.add_option('--maxPage', default=100,    
                  help="the upper bound of pages for department_url")
(options, args) = parser.parse_args()

headers = ('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.86 Safari/537.36')
opener = urllib.request.build_opener()
opener.addheaders=[headers]

fail_log = open(options.failLog, 'w', encoding='utf-8')  
out_file = open(options.outPath, 'w', encoding='utf-8')
related_log = open(options.relatedLog, 'w', encoding='utf-8')
min_len = options.minLen
max_pages = options.maxPage


def process_department_url(department_url, max_pages_num=max_pages):
    """
    Get the experience urls 
    
    Args:
        department_url: such url as https://tag.120ask.com//jingyan/list/1kwm80sksk.html
        max_pages_num: the upper bound of pages for department_url
    Returns:
        the experience urls
    """
    out_urls = []
    for i in range(1, max_pages_num+1):
        try:
            page_url = department_url.replace('.html','')+'-'+str(i)+'.html'
            page_html = opener.open(page_url).read().decode('utf-8') 
            page_soup = BeautifulSoup(page_html, features='lxml')
            lst = page_soup.find_all('li')
            exp_urls = [item.find('a')['href'] for item in lst]
            out_urls += exp_urls
        except:
            break
    return out_urls
    

def process_experience_url(experience_url):
    """
    Process the experience_url and only get the string.
    
    Args:
        experience_url: the url
    Returns:
        the main content,  the related QA content
    """
    html = opener.open(experience_url).read().decode('utf-8') 
    soup = BeautifulSoup(html, features='lxml')
    content = soup.find('div',{'class':'Healthy-left-Summary'}).find_all('p')
    # 把相关问答 和 原始这个网页对应的回答分成两部分
    part_one = '' # experience_url主要内容
    part_two = '' # experience_url 相关问答
    for item in content:
        if item.has_attr('class'):
            part_two += item.string.strip() + '\n'  #这是不同的相关问答，视为不同文档，需要换行
        else:
            string = re.sub(r'[\n\t\s]','',item.string)
            part_one += string   #这是一篇文章，就不换行了
    return part_one, part_two
    

def main(start_url='https://tag.120ask.com/jingyan/'):
    """
    Args:
        start_url: the web url
    """
    
    # 找到呼吸内科、消化内科等对应链接
    url_lst = []
    html = opener.open(start_url).read().decode('utf-8')  
    soup = BeautifulSoup(html, features='lxml')

    lst = soup.find_all('li')
    for item in lst:
        href_lst = [href['href'] for href in item.find_all('a')]
        href_lst = list(set(href_lst))  # 去重
        href_lst = ['https://tag.120ask.com/'+item for item in href_lst]
        url_lst += href_lst
    
    # 对每个科室，得到这个科室对应的所有提问链接
    for department_url in url_lst:
        print(f'Department {department_url} url')
        
        try:
            experience_urls = process_department_url(department_url)     
        except:
            print(f'Department {department_url} can not be processed')
            continue
        
        for exp_url in experience_urls:
            try:
                part_one, part_two = process_experience_url(exp_url)
                out_file.write(part_one + '\n')
                related_log.write(part_two)
            except:
                fail_log.write(exp_url+'\n')
    
    fail_log.close()
    out_file.close()
    related_log.close()
        
    
if __name__=='__main__':
    main()