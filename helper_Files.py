#!/usr/bin/env python
# coding: utf-8

# In[1]:


from helper import *
from helper_View import *
import os


# In[2]:


def files_getFolders(path):
    '''
    returns a list of all folders from the directory
    '''
    all_folders = [x[0] for x in os.walk(path)]
    return all_folders

def files_getFilenamesFromFolder(pathAndTargetFolder):
    '''
    get all fileanmes listed in \path\to\target folder
    eg, pathAndFileName, fileName = files_getFilenamesFromFolder(pathAndTargetFolder)
    '''
    pathAndTargetFolder=pathAndTargetFolder+"\\"
    fileName = []
    pathAndFileName=[]
    for (dirpath, dirnames, filenames) in os.walk(pathAndTargetFolder):
        fileName.extend(filenames)
        # add path to filenames
        for fn in filenames:
            pathAndFileName.append(dirpath+fn)
       
        break    
    return pathAndFileName, fileName

def files_filterFilesWithExtensions(pathAndFilenamesList,extensionList):
    '''get a list of filenames and returns a list of filenames with target extensions'''
    li=[]
    for filename in pathAndFilenamesList:
        fileExtention =  os.path.splitext(filename)[1] # get file extension
        if fileExtention in extensionList:
            li.append(filename)
    return li

def files_renameFiles(pathAndOriginList,newNameList):
    '''
    eg, originalFileNames, pathAndNewList = files_renameFiles(pathAndOriginList,newNameList)
    '''
    originalFileNames = []
    path = os.path.dirname(pathAndOriginList[0]) + "\\"
    pathAndNewList=[]    
    # get original file name
    for fn in pathAndOriginList:
        originalFileNames.append(os.path.splitext(fn)[0])
    # confirmation
    pp(*zip(pathAndOriginList,newNameList))
    answer = input('continue y/n')
    
    if answer =='y':
        for n in range(len(pathAndOriginList)):
            fileExtention = os.path.splitext(pathAndOriginList[n])[1]
            pathAndNewList.append(os.rename(pathAndOriginList[n],
                                           path+newNameList[n]+fileExtention))
    return originalFileNames, pathAndNewList


# In[3]:


def files_overwrite_folders(path,src, dest, ignore_folders_list):
    '''
    copy over src folders to dst. Overwrites same name dest folders. 
    '''
    src=path+src
    dest=path+dest
    
    # if exist, del and re-create
    if os.path.exists(dest):
        shutil.rmtree(dest)
        os.makedirs(dest)
    else:
        os.makedirs(dest)

    # get all folders except ignored
    src_folders = [n for n in os.listdir(src) if not n in ignore_folders_list and os.path.isdir(src+n)]
    print(src_folders)
    # new src_folder_list
    src_folders = [f for f in src_folders]

    for f in src_folders:
        s=src+f
        d=dest+f
        shutil.copytree(s,d)


# In[9]:


# path=r'C:\Users\User\Desktop\musictest'

# one = files_getFolders(path)[1]

# pathFileName, fileName = files_getFilenamesFromFolder(one)
# pathFileName

# files_filterFilesWithExtensions(pathFileName,['.mp3'])

# originalFileNames, pathAndNewList = files_renameFiles(pathFileName,list('abcde'))

# originalFileNames


# In[ ]:




