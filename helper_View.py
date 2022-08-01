#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def view_pandas():
    '''
    -----------------------
    DATAFRAME
    -----------------------
    dictionary:
    d1 = { 'col_employees': ['bob','jack'],                         
               'col_group': ['engrg','acct']}                       
    d2 = {'col1':{'row-a':100, 
                   'row-b':200,
                   'row-c':300}}          
    d3 = [{'col-a':100, 'col-b':200},
          {'col-a':100, 'col-b':200}]  
    list: df6['posts'].values.tolist()
    
    from_dict: 
    a={'k':list('abc'),'m':list('def')}
    b=pd.DataFrame.from_dict(a,orient='columns')   # returns 2 columns, k & m
    -----------------------
    LOCATE
    -----------------------
    df.loc[0:10:2,['age','who']]    
    df.iloc[0:11:2, [1,2]] 
    df.iloc[[0,2,4], [1,2]] 
    df.loc[( df['age'] > 20) & (df['age'] < 40), ['age', 'who']]
    df.at[4, 'B']
    df.iat[2, 3]
    df.isin([0,2])            # find all cells containing. returns bool
    df.isin({'col_A':[0,2]})  # find all values containing values. returns bool
    -----------------------
    MANIPULATE
    -----------------------
    df.apply(np.sqrt) or df.applymap(np.sqrt)    # apply to each cell
    df.apply(np.sum, axis=0)                     # returns column sum of all rows
    -----------------------
    JOIN
    -----------------------
    concat: pd.concat([SeriesA,SeriesB],axis=0) 
    join: dfA.join(dfB)
    merge: pd.merge(df1,df2)                                       # by column  
           pd.merge(df5,df2, left_on='name',right_on='emp_name')   # by column sharing same values
           pd.merge(df3,df4,on='employees',suffixes=['_Lf','_Rt']) # suffix
    rename: df8,rename(columns={'level_0':'users'})
    insert: df.insert(1, "newcol", [99, 99])
    -----------------------
    MULTI-INDEX 
    -----------------------
    pd.DataFrame(values, index=index):
    
    index = pd.MultiIndex.from_tuples([('a',2018),('b',2018),('a',2019),('b',2019)]) 
    values = [[100,200],[300,400],[100,200],[300,400]]
    
    values = [[100,200],[300,400],[500,600],[700,800]]
    index = pd.MultiIndex.from_arrays([['a','a','b','b'], [2018,2019,2018,2019]])
    
    values = [[100,200],[300,400],[150,250],[350,450]]
    index = pd.MultiIndex.from_product([['a','b',], [2018,2019]],names=['area','year'])
    
    Access: run df.__dict__ to locate index and col 
    -----------------------
    GROUBY
    -----------------------
    
    df.groupby('method')['number','mass'].aggregate(['min','max','mean'])
    df.groupby('method').get_group('Radial Velocity')
    
    -----------------------
    PLOT
    -----------------------
    df['pH'].plot(kind='hist')          # specify kind for diff graph
    
    -----------------------
    xx
    -----------------------

    
    
    
    
    
    '''


# In[ ]:


def view_list_comp():
    '''
    consditional: [i for i in range(10) if i%2==0]                    # [0, 2, 4, 6, 8]
    consditional: [i**2 if i %2 == 0 else i **3 for i in range(1, 6)] # [1, 4, 27, 16, 125]
    embedded: 
    [[1,2],[3,4]]     : [b for a in X for b in a] 
    [[[1,2]],[[3,4]]] : [d for a in X 
                        for b in a
                        for c in b
                        for d in c]
    dict & range : {'a': 1, 'b': 2, 'c': 3}
                    {a:n+1 for a,n 
                    in zip(list('abc'),range(3))
                    } 
                        
    
    '''


# In[ ]:


def view_decorator():
    '''
    -----------------------
    CLOSURE - VARIABLES 
    -----------------------
    def wrapper(x):
        def inner(y):
            return x+y
        return inner
    
    main = wrapper(x="X") # SET WRAPPER FUNCTION
    main("Y")             # SET INNER FIUNCTON, RETURNS XY
    -----------------------
    CLOSURE - FUNCTION
    -----------------------
    def wrapper(f,x): 
        def inner(y):
            return f(x) + y
        return inner              # RUN INNER FUNC

    def common(x):
        return x

    main = wrapper(f=common,x="X") # SET WRAPPER FUNCTION
    main("Y")                      # SET INNER FIUNCTON, RETURNS XY
    -----------------------
    DECORATOR - BASIC
    -----------------------
    def wrapper(f):           # MUST INCLUDE (f)
        def inner(*arg):      # GET ARGUMENTS
            res = f(*arg) + "y"
            return res
        return inner

    @wrapper  # TAKES IN BELOW FUNCTION 
    def f(x): # INPUT ARGUMENT
        return x

    f("x")  # RETURNS XY
    -----------------------
    DECORATOR - WITH PARAMS
    -----------------------
    def wrapper(*args, **kwargs):
        print("Inside wrapper")    # FIRST ACT
        def inner(f):
            print(kwargs['inner']) # SECOND ACT
            return f               # THRID ACT
        return inner
 
    @wrapper(inner = "inside inner wrapper")
    def f():
        print("Inside actual function")

    f()
    # RETURNS
        Inside wrapper
        inside inner wrapper
        Inside actual function
    
    '''


# In[ ]:


def view_class():
    '''
    -----------------------
    GET SET DEL
    -----------------------
    def del_age(self):
        del self._age
    age = property(fdel = del_age, fget=get_age, fset=set_age, doc="set and get and del age")
    
    @property
    def pay(self):
        return self._pay
    @pay.setter
    def pay(self, value):
        self._pay = value
    -----------------------
    CLASS STATIC METHOD
    -----------------------
    @classmethod
    def set_new_raise_amt(cls, amt):
        cls.raise_amt = amt              # class variable
        
    @classmethod
    def from_string(cls, string):
        first,last,pay = string.split('-')
        return cls(first,last,int(pay))  # reinstantiate Employee
        
    @staticmethod
    def print_item(x):
        print(x)
    -----------------------
    ENUM
    -----------------------
    from enum import Enum
    import enum
    @enum.unique             # enforce unique values
    class Color(Enum):
        RED = 111
        GREEN = 222
        BLUE = 333
        
        def get_color(self):
            return self.value
        
    Color(111)                # Color.RED
    for c in Color:
        pp(f"{c.value}, {c}") # 111, Color.RED
    c = Color(222)            # Color.GREEN
    c.get_color()             # 222
    -----------------------
    xx
    -----------------------

    
    
    
    '''


# In[130]:


def view_regex():
    '''
    import re
    import regex
    
    def searchgroup(pattern,txt) :
        print(re.search(pattern, txt).groups())
    def findall(pattern,txt,*flag) :
        print(re.findall(pattern, txt,*flag))
    
    
    word: r'Hello'
    word: r'\w+' == r'[a-z]+'            # ['123', 'hello', 'world']
    word: r'[^aeiou]'                    # ['h', 'l', 'l', ' ', 'w', 'r', 'l', 'd', ' ', '1', '0', 'x']
    word: r'[^\w\s]                      # find no words and no space
    digits: r'[0-9]'                     # ['4', '4', '4', '1', '0', '0', '0']
    digits: '\d-0{1}-0{1,3}'             {1} occurence, {1,3} range  
    words+digits+space: '[0-9A-Za-z\s]' 
    condition: 'who|what'
    condition: '(who|what) is'
    boundary: r'\b[a-z]\d+\b'             # a4. left side starting and right side ending 
    ^: re.findall('^what',txt, re.M)      # start sentence. For multi-line 
    $: re.findall('^\w.*\?$',txt, re.M)   # end sentence with "?" For multi-line 
    
    non-greedy starts and moves on, hence it could return multiple items
    txt= 'The Final    End' 
    asterix: r'\w+\s*\w+'                 ['The Final', 'End']  
    plus: r'\w+\s+\w+'                    ['The Final']
    wildcard: r'\w+\s?\w+'                ['The Final', 'End']
    
    greedy only starts and ends once it's done hence only one item in list
    greedy asterix: r'\w+\s.*\w+'         ['The Final    End'] note: .* and .+ operates the same
    greedy wildcard: r'\w+\s.?\w+'        ['The Final']
    
    grouping
    txt='44-sheets-of-a4'
    searchgroup('(\w+)-(\w+)-(\w+)-(\w+)',txt)   returns ('44', 'sheets', 'of', 'a4') grouping based on 
    
    lookahead and lookbehind
    txt = "foobar barfoo". Below all returns 'foo'
    re.search(pattern,txt)
    (?=<lookahead_regex>) look right has: r'foo(?=bar)'   
    (?!<lookahead_regex>) look right doesn't has: r'foo(?!bar)'  
    (?<=<lookbehind_regex>) look left has: r'(?=<bar)foo'
    (?<!<lookbehind_regex>) look left doesn't has: r'(?!<bar)foo'
    
    '''


# In[ ]:


def view_dl():
    '''
    -----------------------
    LAYOUT
    -----------------------
    # DATA 
    # TRAIN TEST SPLIT 
    # MODEL CLASS 
    # TRAIN MODEL 
    # TEST TRAINED MODEL
    # EXPERIMENT & PLOT
    
    -----------------------
    MODEL CLASS
    -----------------------
    class theClassModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.input = nn.Linear(2,8)
            self.hidden = nn.Linear(8,8)
            self.output = nn.Linear(8,3)

        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.hidden(x))
            return self.output(x)
    # TESTING
    model = theClassModel()
    for X,y in train_batches:
        print(model(X[0]))
    -----------------------
    TRAIN THE MODEL
    -----------------------
    def trainTheModel(model):

        # MODEL, LOSSFUN, OPTIM
        model=model
        lossfun=nn.CrossEntropyLoss()
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01)

        # RECORD
        losses=torch.zeros(numepochs)
        trainAcc=[]
        testAcc=[]


        for epoch in range(numepochs):
            # **** TRAINING LOOP ****

            batchLoss=[] # reset after each loop
            batchAcc=[]

            # LOOP EACH BATCH
            for X,y in train_batches:

                # PREDICTION
                yHat=model(X)

                # BACK PROPOGATE
                loss=lossfun(yHat, y)
                dl_back_prop(optimizer, loss)

                # RESULT
                loss=loss.item()
                matches = torch.argmax(yHat,axis=1) == y  
                # RECORD
                batchLoss.append(loss)
                batchAcc.append(100*torch.mean( matches.float()))

            # LOOP EACH EPOCH
            losses[epoch] = np.mean(batchLoss)
            trainAcc.append(np.mean(batchAcc))

            # **** TEST LOOP ****
            model.eval()

            X,y = next(iter(test_batches))
            with torch.no_grad(): # deactivates autograd

                # PREDICTION
                yHat=model(X)
                # RESULT
                matches = torch.argmax(yHat,axis=1) == y  
                testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1)==y).float()) ) 

        return model, trainAcc, testAcc, losses

    
    -----------------------
    Data
    -----------------------
    create data in tensor: torch.tensor(iris.iloc[:,:4].values).float()
    convert single variable: .item()
    convert list variable: .detach()

    -----------------------
    Back Prop
    -----------------------
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    -----------------------
    INFO
    -----------------------
    
    # FIND THE NUMBER OF NODES 
    numNodesInDeep = 0
    for p in deepnet.named_parameters():
        if 'bias' in p[0]:
            pp(p)
            numNodesInDeep += len(p[1])
            
    # FIND THE NUMBER OF TRAINABLE PARAETERS
    nparams = 0
        for p in widenet.parameters():
            if p.requires_grad:
            print('This piece has %s parameters' %p.numel())
            nparams += p.numel()
            
    from torchsummary import summary
    summary(widenet,(1,2))
    -----------------------
    DROPOUT
    -----------------------
    dr = 0.2  # dropoutRate
    
    def __init__(self, dropoutrate):
        super().__init__()

        self.dr = dropoutrate
        self.input  = nn.Linear(2,128)
        self.hidden = nn.Linear(128,128)
        self.output = nn.Linear(128, 1)

    def forward(self,x):
        x = F.relu (self.input(x))
        x = F.dropout(x, p=self.dr, training=self.training) # pp self.training returns True
        x = F.relu(self.hidden(x))
        x = F.dropout(x, p=self.dr, training=self.training) # means to turn off during eval mode

        return self.output(x)

    -----------------------
    BATCH NORMALISE
    -----------------------
    # normalise batch param is from previous layer input

    def __init__(self):

        super().__init__()

        self.input=nn.Linear(11,16)
        self.hidden1=nn.Linear(16,32)
        self.bnorm1 = nn.BatchNorm1d(16) 
        self.hidden2=nn.Linear(32,32)
        self.bnorm2 = nn.BatchNorm1d(32) 
        self.output=nn.Linear(32,1)

    def forward(self,x):
        x=F.relu(self.input(x))

        x= self.bnorm1(x)
        x=F.relu(self.hidden1(x))

        x= self.bnorm2(x)
        x=F.relu(self.hidden2(x))

        return self.output(x)


    -----------------------
    xxx
    -----------------------
    
    
    -----------------------
    xxx
    -----------------------
    
    -----------------------
    xxx
    -----------------------
    
    
    -----------------------
    xxx
    -----------------------
    '''


# In[ ]:


def view_general():
    '''
    -----------------------
    ARG
    -----------------------
    def args (*a):
        pp(a)
    args(1,2,3)
    
    -----------------------
    KWARG
    -----------------------
    def color_size_1 (**kw):
        pp(kw)
    color_size_1(color='red',size=20) 
    
    def color_size_2 (**kw):
        pp(f"color is {kw['color']}, size is {kw['size']}") 
    params = {'color':'red','size':20}
    color_size_2(**params)
    
    -----------------------
    POSITIONAL
    -----------------------
    def positionalKW(a1, a2, *a, option=True, **kw):
    
    -----------------------
    LAMBDA
    -----------------------
    lambda <args> : <return Value> if <condition > ( <return value > if <condition> else <return value>)
    a = lambda x : x                                       # a(10)  10
    b = (lambda x: x>2) (3)                                # b      True
    c = (lambda x, y: x ** 2 + y ** 2) (2, 3)              # c      13
    d = (lambda x, y: x ** 2 + y ** 2)                     # d(3,5) 34
    e = lambda x : True if (x > 10 and x < 20) else False  # e(12)  True
    
    -----------------------
    MAP, FILTER, REDUCE
    ----------------------- 
    li = [12, 65, 54, 39, 102, 339, 221, 50, 70 ]
    list(filter(lambda x: (x % 13 == 0), li)) 
    reduce(lambda x,y:x+y, li)
    list(map(lambda x: x*x, li))
    
    -----------------------
    xx
    -----------------------
    
    -----------------------
    xx
    -----------------------
    
    -----------------------
    xx
    -----------------------
    
    -----------------------
    xx
    -----------------------
    
    -----------------------
    xx
    -----------------------
    
    -----------------------
    xx
    -----------------------
    
    -----------------------
    xx
    -----------------------
    '''
    


# In[ ]:





# In[ ]:




