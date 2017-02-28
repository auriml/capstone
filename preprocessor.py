from gensim import logging
from gensim.models import Doc2Vec ,Phrases
from gensim.models.doc2vec import TaggedDocument


from num2words import num2words
import re
import os
import pandas as pd



dataDirectory  =  '/Users/aureliabustos/Downloads/search_result/'
#dataDirectory  =  './sampleData/'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def retrieve_info(path, tags):

    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    text = {}
    try:
        for  tag in tags:
            if tag == "eligibility" :
                text[tag]= root.find('eligibility').find('criteria').find('textblock').text
            if tag ==  "intervention_name":
                intBrowse = root.find('intervention_browse')
                if  intBrowse:
                    text[tag]= [i.text for i in intBrowse.findall('mesh_term') ]
                else:
                    text[tag] = [i.text for i in root.iter('intervention_name') ]
            if tag == "condition" :
                text[tag]=  [c.text for c in root.findall('condition') ]
            if tag == "enrollment":
                text[tag]= root.find('enrollment').text
    except:
        print ("no eligibility criteria")

    return text

# sentence splitter
alteos = re.compile(r'([!\?:])')
decimals = re.compile(r'((\d)\.(\d))')
abbrev = re.compile(r'(\w)\.(\w)\.?')
bullet = re.compile(r'(\W+\d+\.[^\n\w]|\n\n|\W-\W)')

def sentencesSplitter(l):
    l = l.lower()
    l = decimals.sub('\1 doc \2',l)
    l = abbrev.sub('\1\2',l)
    l = bullet.sub('.',l)
    l = alteos.sub('.', l)
    sentences = re.compile(r'\.+\W*').split(l)
    sentences = list(filter(('').__ne__, sentences))
    return sentences

def sentencesWithConditions(sentences, conditions):
    lines = []
    cancerConditions =  selectOnlyCancerConditions(conditions)
    for s in sentences:
        if isCancer(s):
            lines.append( s)
        else:
            for c in cancerConditions:
                lines.append(c + ' diagnosis and ' + s)
    return lines

def isCancer(s):
    return any(ext in s.lower() for ext in ["cancer" , "neoplasm" , "oma", "tumor"])

def selectOnlyCancerConditions(conditions):
    filteredList = []
    for c in conditions:
        if isCancer(c):
            filteredList.append(c )

    return filteredList


# cleaner (order matters)
def clean(text, convertnum2words = False, removeSingles = False):
    # positive
    pos =  re.compile(r'\++', re.U)
    # negative
    neg =  re.compile(r'\w-\W', re.U)
    contractions = re.compile(r"'|-|\"|/|\(|\)|,")
    # all non alphanumeric
    symbols = re.compile(r'\W+', re.U)
    # single character removal
    singles = re.compile(r'(\s\S\s)', re.I|re.U)
    # separators (any whitespace)
    seps = re.compile(r'\s+')
    # all numeric
    digits = re.compile(r'\d+')
    #comparatives
    equalThan =   re.compile(r'=')
    greaterThan = re.compile(r'>|≥')
    lessThan = re.compile(r'<|≤')

    text = text.lower()
    text = pos.sub(" positive ", text)
    text = neg.sub(" negative ", text)
    text = contractions.sub(' ', text)
    text = greaterThan.sub('greater_than ', text)
    text = lessThan.sub('less_than ', text)
    text = equalThan.sub('equal_than ', text)
    text = symbols.sub(r' ', text)
    if removeSingles:
        text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    text = text.strip()
    if convertnum2words:
        for d in digits.findall(text):
            text = text.replace(d, num2words(int(d)) )
    return text

class MySentences(object):
    def __init__(self, dirname, bigrams):
        self.dirname = dirname
        self.bigrams = bigrams

    def __iter__(self):
        for filename in os.listdir(self.dirname):
            if filename.endswith(".xml"):
                path = os.path.join(self.dirname, filename)
                eligibility = retrieve_info(path, ['eligibility','intervention_name', 'condition'])
                if eligibility.__len__() > 0:
                    for uid,  line  in enumerate(sentencesWithConditions(sentencesSplitter(eligibility['eligibility']), eligibility['condition'])):
                        line = clean(line, convertnum2words=True, removeSingles=False)
                        if self.bigrams:
                            line =  self.bigrams[line.split()]
                            yield line
                        else:
                            yield line.split()



#generate bigrams and save them
def generateBigrams(sentences):
    bigram_transformer = Phrases(sentences, min_count=20, threshold=500)
    bigram_transformer.save("bigrams", pickle_protocol=3)

    fd = open("bigrams.txt", 'a')
    for phrase, score in bigram_transformer.export_phrases(sentences):
        fd.write(u'{0}   {1}'.format(phrase, score))
    fd.close()

    return bigram_transformer


#generate a file of plain utf words separated by a single space needed as input to wordembeddings
def text2words_to_csv(dataDirectory, fname, bigrams = True):

    for filename in os.listdir(dataDirectory):
        if filename.endswith(".xml"):
            fd = open(fname,'a')
            path = os.path.join(dataDirectory, filename)
            eligibility = retrieve_info(path, ['eligibility'])
            if eligibility.__len__() > 0:
                for line  in sentencesSplitter(eligibility['eligibility']):
                    line = clean(line, convertnum2words=True, removeSingles=False)
                    if bigrams:
                        line =  bigrams[line.split()]
                    fd.write(" ".join(line) + " ")

            fd.close()

#generate processed eligibility criteria sentences classified by exclusion/inclusion and by treatment
#optional add conditions to criterion
def to_csv(fname, bigrams = False, conditions=False,  fields=['eligibility','intervention_name','condition']):



    keys = False
    for filename in os.listdir(dataDirectory):
        print(filename)
        if filename.endswith(".xml"):
            path = os.path.join(dataDirectory, filename)
            data = retrieve_info(path, fields)
            if not 'eligibility' in data:
                continue

            if conditions:
                criteria = sentencesWithConditions(sentencesSplitter(data['eligibility']),data['condition'])
            else:
                criteria = sentencesSplitter(data['eligibility'])



            if criteria.__len__() > 0:
                fd = open(fname,'a')
                if keys is False:
                    string = 'eligible'+ '\t' + 'intervention_name' + '\t' + 'eligibility'
                    fd.write(string + '\n')
                    keys = True
                eligible = True
                for criterion in criteria:
                    line = clean(criterion, convertnum2words = True, removeSingles=False)
                    if line.__contains__("inclusion criteria"):
                        eligible = True
                        continue
                    elif line.__contains__("exclusion criteria"):
                        eligible = False
                        continue
                    if line == "" :
                        continue

                    eligibleCriterion = ""
                    if line.startswith("no "):
                        eligibleCriterion = "False"
                        line = line.replace("no ","")
                    if eligibleCriterion  is  "":
                        eligibleCriterion = str(eligible)

                    #insert bigrams
                    if bigrams:
                        line =  bigrams[line.split()]
                        line = ' '.join(line)

                    # single character removal
                    singles = re.compile(r'(\s\S\s)', re.I|re.U)
                    line = singles.sub(' ', line)

                    for i in data['intervention_name']:

                        values = []
                        string = ""
                        separator = '\t'

                        values.append(eligibleCriterion)
                        values.append(i)
                        values.append(line)


                        string = separator.join(values) + '\n'
                        fd.write(string)

                fd.close()

            continue
        else:
            continue

def nciThesaurusNER(source_csv = './textData/data.csv', fname = './textData/intervention_index.csv' ,field = 'intervention_name'):
     import requests
     dict = {}
     print('Loading source dataset')
     df = pd.read_csv(source_csv, sep='\t', header=0)
     # dictionary mapping label name to numeric id
     values = df[field].unique()
     dictIndex = {k: v for v, k in enumerate(values)}
     fd = open(fname,'a')
     fd.write("key" + '\t' + "index"+ '\t' + "value" + '\n')
     for k in dictIndex:
         keys = re.compile(r'\(|\)').split(k)
         for key in keys:
             try:
                 link = "http://nlp.medbravo.org/c.groovy?concept=" + key
                 f = requests.get(link)
                 f.encoding = "utf-8"
                 d = f.json()

                 mappedConcepts = d['results']['resolvedConcepts']
                 if not mappedConcepts:
                     if len(key.split()) > 2:
                        link = "http://nlp.medbravo.org/p.groovy?phrase=" + key
                     else:
                        link = "http://nlp.medbravo.org/c.groovy?concept=" + key

                     f = requests.get(link)
                     f.encoding = "utf-8"
                     d = f.json()
                     mappedConcepts = d['results']['resolvedConcepts']


                 paths = d['results']['facets']
                 filteredPaths =  [p for p in paths if any(m in p for m in mappedConcepts)]
                 dict[key]=[dictIndex[key],filteredPaths]
                 string =  key + '\t' + str(dictIndex[key])  + '\t' + ','.join(filteredPaths) + '\n'
                 fd.write(string)
             except:
                 print(key)
                 continue
     fd.close()


     return dict

def toFastText_format(source_csv = './textData/dataWithCondition.csv', fname = "./textData/dataFastText.csv" , labeledField = "eligible", otherFields = ["condition","intervention_name", "eligibility" ]):
    #PRECONDITION: it requires the list of all possible values of labeledField in the data
    print('Loading source dataset')
    df = pd.read_csv(source_csv, sep='\t', header=0)
    # dictionary mapping label name to numeric id
    labels_values = df[labeledField].unique()
    labels_index = {k: v for v, k in enumerate(labels_values)}

    separator = ' . '

    fd = open(fname,'a')
    for row in df.iterrows():
        string = ""
        vals = []
        labeledValue = row[1][labeledField]
        for field in otherFields:
            if field == "condition" and field in df:
                vals.append('patients diagnosed with '  + ' or '.join(row[1][field]))
            if field == "intervention_name" and field in df:
                vals.append('study interventions are ' + row[1][field])
            if field == "eligibility" and field in df:
                vals.append(row[1][field])
            if field == "eligible" and field in df:
                string = str(row[1][field]) + '\t'
        string = string +  '__label__' + str(labels_index[labeledValue])  +  '\t' + separator.join(vals) + '\n'
        fd.write(string)
    fd.close()


def appendProcesssedFieldToCsv(source_csv = './textData/data.csv', fname = './textData/dataWithInterventionClass.csv',  dic_csv ='./textData/intervention_index.csv', new_fieldName = 'intervention_class' , source_fieldName = 'intervention_name'):
    print('Loading source dataset')
    df = pd.read_csv(source_csv, sep='\t', header=0)
    print('Loading dicccionary for field')
    dicf = pd.read_csv(dic_csv, sep='\t', header=0)
    keys = dicf.key
    values = dicf.value
    dic = dict(zip(keys, values))
    df[new_fieldName] = df[source_fieldName].map(lambda x: dic.get(x) if dic.get(x) else 'UNK')
    df.to_csv(sep='\t', path_or_buf=fname)



#generate bigrams
print("Starting to generate bigrams")
#sentences = MySentences(dataDirectory,bigrams=False)
#bigram = generateBigrams(sentences)
#bigram = Phrases.load("bigrams")
#line = "I have myasthenia gravis and take trimethoprim sulfamethoxazole".split()
#print(bigram[line])

#extracts criteria by treatment and eligibility (replace bigrams)
print("Starting to process eligibility criteria and extracts criteria sentences by treatment and eligibility (replace bigrams)")
#to_csv("./textData/data.csv", bigrams = bigram )

#extracts criteria by  treatment, eligibility and conditions (replace bigrams)
print("Starting to process eligibility criteria and extracts criteria sentences by treatment, eligibility and conditions (replace bigrams)")
to_csv("./textData/dataWithConditions_no_bigrams.csv", bigrams = False, conditions = True )

# nciNER for intervention_name field
print("Starting to do NER for intervention_name using nciThesaurus ")
#nciThesaurusNER(source_csv = './textData/data.csv', fname = './textData/intervention_index.csv' ,field = 'intervention_name')

# append intervention_class to data
print("Appending nci intervention classes to data")
#appendProcesssedFieldToCsv(source_csv = './textData/dataWithConditions.csv', fname = './textData/dataWithInterventionClass.csv',  dic_csv ='./textData/intervention_index.csv', new_fieldName = 'intervention_class' , source_fieldName = 'intervention_name')


#generate labeled criteria with FastText format
print("Generate labeled criteria from processed criteria with FastText format")
toFastText_format(source_csv = './textData/dataWithConditions_no_bigrams.csv', fname = "./textData/labeledEligibilityFastText_no_bigrams.csv",labeledField = "eligible")
#toFastText_format(source_csv = './textData/dataWithConditions.csv', fname = "./textData/labeledInterventionFastText.csv", labeledField = "intervention_name", otherFields = ["condition","eligibility", "eligible" ])
#toFastText_format(source_csv = './textData/dataWithInterventionClass.csv', fname = "./textData/labeledInterventionClassFastText.csv", labeledField = "intervention_class", otherFields = ["condition","eligibility", "eligible" ])


#generate plain words file needed for wordembeddings
print("Starting to generate a file containing all criteria transformed in a unique sequence of utf8  words separated by spaces ")
text2words_to_csv(dataDirectory, "./textData/words_data_no_bigrams.csv", bigrams = False)



