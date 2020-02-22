# -*- coding: utf-8 -*-
"""
相关报道咨询 https://www.cmtopdr.com/post/
"""
from bs4 import BeautifulSoup
from optparse import OptionParser
import urllib


parser = OptionParser()
parser.add_option("--failLog", default="Report_fail_log.txt",
                  help="the file to record the web which can not be parsed", metavar="FILE")
parser.add_option("--outPath", default="Report_out.txt",
                  help="the output file name ")
(options, args) = parser.parse_args()


headers = ('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.86 Safari/537.36')
opener = urllib.request.build_opener()
opener.addheaders=[headers]


fail_log = open(options.failLog, 'w', encoding='utf-8')  
out_file = open(options.outPath, 'w', encoding='utf-8')


def process_one_report(report_url='https://www.cmtopdr.com/post/detail/5b1a2389-2338-41be-9886-b32327160ed7'):
    """
    Args:
        report_url: the url of clinical report/news...
    Returns:
        the parsed string
    """
    html = opener.open(report_url).read().decode('utf-8')  # if has Chinese, apply decode()
    soup = BeautifulSoup(html, features='lxml')

    lst = soup.find_all('div', {'class': 'w'})
    content = lst[-1].contents
    content = content[2:]   #  第一行是空行，第二行是作者，跳过
    content_lst = list(filter(lambda s: s!=None, [item.string for item in content]))  # 提前字符，并把非字符的过滤掉
    content_str = ' '.join(content_lst).replace('\n','').replace('\xa0','').strip()
    return content_str


def main(start_url='https://www.cmtopdr.com/post/', max_pages=float('inf')):
    """
    Args:
        start_url: the web url
        max_pages: the maximum pages to parse. Default is infinity.
    """
    url = start_url
    page_id = 0

    while True:
        if page_id >= max_pages:
            break
        print(f'now pages {page_id}')
        html = opener.open(url).read().decode('utf-8')
        soup = BeautifulSoup(html, features='lxml')
        news_list = soup.find('div', {'class': 'news_list'})
        news_list = news_list.find_all('a')
        for item in news_list:
            if 'attrid' in item.attrs:    # 说明这是资讯，应该保存下来
                news_url = 'https://www.cmtopdr.com' + item.attrs['href']
                if 'post' in news_url:   # 解析资讯相关网页
                    try:
                        news_string = process_one_report(news_url)
                        out_file.write(news_string+'\n')
                    except:
                        fail_log.write(news_url+'\n')
            else:   # 页面数字 下一页按钮
                if item.contents == ['>']:
                    url = start_url + item.attrs['href']   # 下一页的链接
        page_id += 1


if __name__ == '__main__':
    main(start_url='https://www.cmtopdr.com/post/', max_pages=3000)
    out_file.close()
    fail_log.close()
