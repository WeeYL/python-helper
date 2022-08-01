#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyautogui
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException        

import requests
from bs4 import BeautifulSoup

from helper import *


# In[2]:


class Selenium_Class:
    '''
    sel = Selenium_Class(url)
    sel.get_driver()
    '''
    chrome_path=r"C:\Program Files (x86)\Google\Chrome\Application"
    
    def __init__(self, url):
        self.url = url
        
    '''CONFIG'''
    def get_driver(self): 
        '''setup webdriver and return webdriver'''
        chromer_driver_path = "C://Users//User//1.programming//1.tutorial//webdriver//chromedriver.exe"
        self.driver = webdriver.Chrome(chromer_driver_path) 
        self.driver.get(self.url)
        return self.driver

    # config
    def get_chrome_profile_driver(self):
        r'''
        Open terminal 
        C:\Program Files (x86)\Google\Chrome\Application
        chrome.exe --remote-debugging-port=9222 --user-data-dir=C:\Users\User\Desktop\chromeprofile
        '''
        opt=Options()
        chromedriver_path="C:\\Users\\User\\1.programming\\1.tutorial\\webdriver\\chromedriver.exe"
        # run url
        opt.add_experimental_option("debuggerAddress","localhost:9222")
        self.driver=webdriver.Chrome(executable_path=chromedriver_path,options=opt)
        self.driver.get(self.url)
        return self.driver
    
    
    '''WEB ELEMENTS'''
    def get_available_attributes_in_css(self,css_selector):
        '''takes in a css selector and returns all the attributes in the css
        eg, sel.get_available_attributes_in_css('td')
        eg, sel.get_available_attributes_in_css('.form-control.stake-input'))
        '''
        elements = self.driver.find_elements(By.CSS_SELECTOR, css_selector)
        attrs = [self.driver.execute_script('var items = {}; for (index = 0; index < arguments[0].attributes.length; ++index) { items[arguments[0].attributes[index].name] = arguments[0].attributes[index].value }; return items;', element) for element in elements]      
        attrs_list = []
        for n in range(len(attrs)):
            if ((attrs[n] not in attrs_list) and ( attrs[n] != {}) ):
                attrs_list.append(attrs[n])
        return attrs_list
    
    def get_available_attribute_values(self,css_selector="a", attribute="href" ):
        '''eg webel_list = get_css_and_attribute('a','href')'''
        web_elems = self.driver.find_elements(By.CSS_SELECTOR, css_selector)
        webel_list=[]
        for elem in web_elems:
            webel_list.append(elem.get_attribute(attribute))
        webel_list = set([e for e in webel_list if e != None if e != ""])
        return webel_list

    
    def check_exists_by_link_text(text):
        '''returns boolean'''
        self.text=text
        try:
            self.driver.find_element(By.PARTIAL_LINK_TEXT,self.text)
        except NoSuchElementException:
            return False
        return True

    def get_elements_by_attribute(self,css_selector,attribute,attribute_val):
        '''returns web element of attribute'''
        self.css_selector=css_selector 
        self.attribute=attribute
        self.attribute_val=attribute_val
        elements=self.driver.find_elements(By.CSS_SELECTOR,self.css_selector)
        li = [element for element in elements if  element.get_attribute(self.attribute)==self.attribute_val]
        return li 
    
    
    '''CLICK '''
    def click_by_elem_and_attributes(self,driver,css_selector,attribute, attribute_val):
        '''
        <input class="form-control stake text-right pull-right stake-input" 
        type="text" autocomplete="off" min="0" step="0.01" max="99999.99" value=""
        >
        
        eg, sel.click_by_elem_and_attributes(sel,'.form-control.stake-input','type','text')
        '''
        self.css_selector=css_selector
        self.attribute=attribute
        self.attribute_val=attribute_val
        

        elements=self.driver.find_elements(By.CSS_SELECTOR,self.css_selector)
        print(elements)
        self.attribute_val=str(self.attribute_val)
        for element in elements:
            if element.get_attribute(self.attribute)==self.attribute_val:
                try: 
                    element.click()
                    break
                except:
                    continue
    def click_after_verify(self,selector_type, value, wait=2):
        '''
        wait and click selector 
        selector type: \ncs: css_selector, 
        plt: partial_link_text
        '''
        self.selector_type=selector_type
        self.value=value
        
        if self.selector_type=="cs":
            res = WebDriverWait(self.driver, wait).until(EC.presence_of_element_located((By.CSS_SELECTOR, self.value)))
        elif self.selector_type=="plt":
            res = WebDriverWait(self.driver, wait).until(EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, self.value)))
        res.click()
    
    '''BROSWER'''
    def launch_href_elem_in_new_tab(self,web_elem):
        '''elem are result from driver.find_element'''
        self.web_elem = web_elem
        self.web_elem.send_keys(Keys.CONTROL + Keys.RETURN) 
    
    def close_all_tabs_except(self,num):
        self.num = num
        '''close pop ups'''
        while len(self.driver.window_handles) > self.num:
            self.driver.switch_to.window(self.driver.window_handles[-1]) # switch to last tab
            self.driver.close()
        self.driver.switch_to.window(self.driver.window_handles[-1]) # switch to main tab



# In[3]:


class Soup_Base:
    
    def __init__(self,url):
        '''After init, run get_soup()'''
        self.url=url
        
    def get_soup(self):
        response = requests.get(url=self.url)
        result = response.content
        self.soup = BeautifulSoup(result,'html.parser')
        return self.soup
    
    def soup_get_tag (self,css):
        '''eg, browser.soup_get_css('td')'''
        self.css = css
        li_soup = self.soup.find_all(self.css)
        return li_soup
    
    def soup_get_attrbutes(self,css,attribute):
        '''
        soup returns all the values of the attributes in the css
        eg, mySoup.soup_get_attrbutes('td','data-stat')
        '''
        self.css = css
        self.attribute = attribute
        li =[]
        li_soup = self.soup.find_all(self.css)
        for link in li_soup:
            li.append(link.get(self.attribute))
        return set(li)

    def soup_get_attrbutes_by_value(self,css,attribute_dict):
        '''eg, soup_get_find_all.find_all('td',{'data'="goals_for"})'''
        self.css = css
        self.attribute_dict = attribute_dict
        return self.soup.find_all(self.css,self.attribute_dict)


# In[ ]:





# In[ ]:




