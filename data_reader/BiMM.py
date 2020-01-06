# -*- coding: utf-8 -*-
"""
双向最大匹配
"""
class forward_MM():
    '''正向最大匹配'''
    def __init__(self, dic_set, dic_class_name, word_max_len=20):
        ''' word_max_len：最大匹配时搜索的最大长度上限'''
        self.dic_set = dic_set  #词典的类型是set
        self.dic_class_name = dic_class_name  #比如是手术类、药物类
        self.max_len = min(  max([len(w) for w in dic_set]), word_max_len)
      
    def cut(self, string):
        cut_lst = []
        
        start = 0
        end = 0 
        while end<len(string):
        
            for end in range(  min(start+ self.max_len, len(string)),  start,  -1):   
                s = string[start:end]
                if s in self.dic_set:   
                    cut_lst.append(s)
                    break
                if len(s) == 1:
                    cut_lst.append(s)
            start = end
        return cut_lst
            

class backward_MM():
    '''逆向最大匹配'''
    def __init__(self, dic_set, dic_class_name, word_max_len=20):
        ''' word_max_len：最大匹配时搜索的最大长度上限'''
        self.dic_set = dic_set  #词典的类型是set
        self.dic_class_name = dic_class_name  #比如是手术类、药物类
        self.max_len = min(  max([len(w) for w in dic_set]), word_max_len)
       
    def cut(self, string):
        cut_lst = []
        
        start = len(string)
        end = len(string)
        while start>0:
        
            for start in range(  max(end-self.max_len,0), end, 1):
                s = string[start:end]
                if s in self.dic_set:   
                    cut_lst.append(s)
                    break
                if len(s) == 1:
                    cut_lst.append(s)
            end = start
        return cut_lst[::-1]            
         
        
        
class BiMM():
    def __init__(self, dic_set, dic_class_name, word_max_len=20):
        ''' word_max_len：最大匹配时搜索的最大长度上限'''
        self.fmm = forward_MM(dic_set, dic_class_name, word_max_len)
        self.bmm = backward_MM(dic_set, dic_class_name, word_max_len)
        self.dic_class_name = dic_class_name
        self.dic_set = dic_set
    
    def cut(self,string):
        fmm_res = self.fmm.cut(string)
        bmm_res = self.bmm.cut(string)
        return fmm_res if len(fmm_res)<=len(bmm_res) else bmm_res
    
    

if __name__ == '__main__':
    dic_set = {'今天','天气','怎么样'}
    string = '我们问今天天气怎么样'
    
    BiMM_class = BiMM(dic_set, 'sth')
    cut_lst = BiMM_class.cut(string)
    
    print(cut_lst)
    
