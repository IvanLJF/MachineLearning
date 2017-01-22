import pandas as pd
import sys
from math import log
#2.7.12 |Anaconda 4.0.0 (64-bit)|

print (sys.version)


#Step 1 - Read Training data
input_file = "Training.xls"
df = pd.read_csv(input_file,header=0)

#Step 2 - Replace with numerical values
d = {'Male': 1, 'Female':2}
df['Gender'] = df['Gender'].map(d)
#Remove insignificant id column
df.drop(['Patient_ID'],1,inplace=True)
#Remove Unnamed Column
for col in df.columns:
    if 'Unnamed' in col:
        del df[col]
HPV = {'Positive': 1, 'Negative':-1}
df['HPV_p16_status'] = df['HPV_p16_status'].map(HPV)
race = {'White': 1, 'Black':2, 'Hispanic':3, 'Asian':4}
df['Race'] = df['Race'].map(race)
t_side = {'L': 0, 'R':1}
df['Tumor_side'] = df['Tumor_side'].map(t_side)
t_sub_site = {'Tonsil': 0, 'BOT':1, 'Other':2, 'Pharyngeal_wall':3, 'GPS':4, 'Soft_palate':5}
df['Tumor_subsite'] = df['Tumor_subsite'].map(t_sub_site)
n_category =  {'0':0,'1':1,'2a':2,'3':3,'2c':4,'2b':5}
df['N_category'] = df['N_category'].map(n_category)     
ajcc_stage =  {'II': 0,'III':1,'IV':2,'I':3}
df['AJCC_Stage'] = df['AJCC_Stage'].map(ajcc_stage)     
path_grade = {'III':0, 'II':1, 'NA':2,  'I':3, 'II-III':4, 'IV':5}
df['Pathological_grade'] = df['Pathological_grade'].map(path_grade)     
smok_status = {'Former':1, 'Current':2, 'Never':3,'NA':4}
df['Smoking_status_at_diagnosis'] = df['Smoking_status_at_diagnosis'].map(smok_status)     
therapy_unique = {'N':1,'Y':2}
df['Induction_Chemotherapy'] = df['Induction_Chemotherapy'].map(therapy_unique)     
df['Concurrent_chemotherapy'] = df['Concurrent_chemotherapy'].map(therapy_unique)  
df = df.fillna(-99)
features = list(df.columns)

df.to_csv('Training_new.xls', sep=',', header = False, index = False)
my_data=[line.strip().split(',') for line in file('Training_new.xls')]

#Step 3 - Dividing sets
def divideset(rows,column,value):
   #Make a function for two groups of datasets
   split_function=None
   if isinstance(value,int) or isinstance(value,float): # check if the value is a number i.e int or float
      split_function=lambda row:row[column]>=value
   else:
      split_function=lambda row:row[column]==value
   # Divide the rows into two sets and return them
   set1=[row for row in rows if split_function(row)]
   set2=[row for row in rows if not split_function(row)]
   return (set1,set2)
   
#Step 4 - Create counts of possible results (the last column of each row is the result)
def uniquerowcounts(rows):
   results={}
   for row in rows:
      # The result is the last column
      r=row[len(row)-1]
      if r not in results: results[r]=0
      results[r]+=1
   return results
   
#Step 5 - Sum all the values and use the average value for split
def computaveragevalueforsplit(rows):
   results_list = []
   for row in rows:
       results_list += row
   sum = 0.0
   ent=0.0
   for i in range(1,len(results_list)):
       sum+=float(results_list[i])
   if(len(results_list)>0):
       ent=float(sum/len(results_list))
       return ent
   return ent

#Step 6 - Entropy Shanon entropy
def customshanonentropy(rows):
   log2=lambda x:log(x)/log(2)  
   results=uniquerowcounts(rows)
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(rows)
      ent=ent-p*log2(p)
   return ent

   
#col- column index criteria
#value - value column must match
#tb, fb - decision nodes
#results - stores a dictionary of results
class decisiontreenode:
  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
    self.col=col
    self.value=value
    self.results=results
    self.tb=tb
    self.fb=fb

#Step 7 - build tree based on average value of columns (features)
def buildtreeaveragevalueforsplit(rows,scoref=computaveragevalueforsplit): 
  if len(rows)==0: return decisiontreenode()
  current_score=scoref(rows)
  best_gain=0.0
  best_criteria=None
  best_sets=None
  column_count=len(rows[0])-1   #count the # of attributes/columns. 
  for col in range(0,column_count):
    column_values={}            
    for row in rows:
       column_values[row[col]]=1   
    for value in column_values.keys():
      (set1,set2)=divideset(rows,col,value) 
      # Information gain
      p=float(len(set1))/len(rows) #p is the size of a child set relative to its parent
      gain=current_score-p*scoref(set1)-(1-p)*scoref(set2) #cf. formula information gain
      if gain>best_gain and len(set1)>0 and len(set2)>0: #set must not be empty
        best_gain=gain
        best_criteria=(col,value)
        best_sets=(set1,set2)
  if best_gain>0:
    trueBranch=buildtreeaveragevalueforsplit(best_sets[0])
    falseBranch=buildtreeaveragevalueforsplit(best_sets[1])
    return decisiontreenode(col=best_criteria[0],value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
  else:
    return decisiontreenode(results=uniquerowcounts(rows))

#Step 8 - build tree based on shanon entropy
def buildtreeentropy(rows,scoref=customshanonentropy): 
  if len(rows)==0: return decisiontreenode()
  current_score=scoref(rows)
  best_gain=0.0
  best_criteria=None
  best_sets=None
  column_count=len(rows[0])-1   #count the # of attributes/columns. 
  for col in range(0,column_count):
    column_values={}            
    for row in rows:
       column_values[row[col]]=1   
    for value in column_values.keys():
      (set1,set2)=divideset(rows,col,value) 
      # Information gain
      p=float(len(set1))/len(rows) #p is the size of a child set relative to its parent
      gain=current_score-p*scoref(set1)-(1-p)*scoref(set2) #cf. formula information gain
      if gain>best_gain and len(set1)>0 and len(set2)>0: #set must not be empty
        best_gain=gain
        best_criteria=(col,value)
        best_sets=(set1,set2)
  if best_gain>0:
    trueBranch=buildtreeentropy(best_sets[0])
    falseBranch=buildtreeentropy(best_sets[1])
    return decisiontreenode(col=best_criteria[0],value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
  else:
    return decisiontreenode(results=uniquerowcounts(rows))

#Step 9 - print decision tree
def printdecisiontree(tree,indent=''):
   if tree.results!=None:
      print str(tree.results)
   else:
      print str(tree.col)+':'+str(tree.value)+'? '
      print indent+'T->',
      printdecisiontree(tree.tb,indent+'  ')
      print indent+'F->',
      printdecisiontree(tree.fb,indent+'  ')

#Step 10 - prune tree based on shanon entropy
def prunedecisiontree(tree,mingain, evaluationFunction=customshanonentropy, notify=False):
    # If the branches aren't leaves, then prune them
    if tree.tb.results==None:
        prunedecisiontree(tree.tb,mingain,evaluationFunction, notify)
    if tree.fb.results==None:
        prunedecisiontree(tree.fb,mingain,evaluationFunction, notify)
    # If both the subbranches are now leaves, see if they
    # should merged
    if tree.tb.results!=None and tree.fb.results!=None:
        # Build a combined dataset
        tb,fb=[],[]
        for v,c in tree.tb.results.items( ): tb+=[[v]]*c
        for v,c in tree.fb.results.items( ): fb+=[[v]]*c
        p = float(len(tb)) / len(tb + fb)
        delta = evaluationFunction(tb+fb) - p*evaluationFunction(tb) - (1-p)*evaluationFunction(fb)
        # Test the reduction in entropy
        if delta<mingain:
            # Merge the branches
            tree.tb,tree.fb=None,None
            tree.results=uniquerowcounts(tb+fb)

#Step 11 - Classify Decision Tree
def classify(observation,tree):
    if tree.results!=None:
        return tree.results
    else:
        v=observation[tree.col]
    branch=None
    if isinstance(v,int) or isinstance(v,float):
        if v>=tree.value: branch=tree.tb
        else: branch=tree.fb
    else:
        if v==tree.value: branch=tree.tb
        else: branch=tree.fb
    return classify(observation,branch)    

#Step 12 - Classify with missing data
def classifyWithMissingData(observation,tree):
    if tree.results!=None:
        return tree.results
    else:
        v=observation[tree.col]
    if v==None:
        tr = classifyWithMissingData(observation,tree.tb)
        fr = classifyWithMissingData(observation,tree.fb)
        tcount=sum(tr.values( ))
        fcount=sum(fr.values( ))
        tw=float(tcount)/(tcount+fcount)
        fw=float(fcount)/(tcount+fcount)
        result={}
        for k,v in tr.items( ): result[k]=v*tw
        for k,v in fr.items( ): result[k]=v*fw
        return result
    else:
        if isinstance(v,int) or isinstance(v,float):
            if v>=tree.value: branch=tree.tb
            else: branch=tree.fb
        else:
            if v==tree.value: branch=tree.tb
            else: branch=tree.fb
        return classifyWithMissingData(observation,branch)

print('build tree optimal split computed on average value')
tree=buildtreeaveragevalueforsplit(my_data)
printdecisiontree(tree)

print('classify function call')
print(classify('[0,1,1,58,1,0,0,2,0,0,0.0,1,5.0,41,66,30,1,1,1]',tree))

print('===========')
print('Tree based on entropy computation')
tree=buildtreeentropy(my_data)
printdecisiontree(tree)

print('prune - gain value >  .01')
#minimum gain value to prune
prunedecisiontree(tree,.01)        
printdecisiontree(tree)

print('With Missing Data')
print(classifyWithMissingData('[0,1,None,58,1,0,0,2,0,0,0.0,1,5.0,41,66,30,1,1,1]',tree))
