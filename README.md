# Dualingual feedback comments topic modeling
Topic modeling with LDA, CTM and STM on RMarkdown document
---
title: "Customers Feedbacks Topic Modeling"
author: "Trung Nguyen"
date: "10 December, 2018"
output: 
  html_document: 
    toc: true
    toc_float: true
    theme: united
    number_section: true
    toc_depth: 3
    highlight: tango
    keep_md: true
---



__Timespan:__ 1 week

In this mini project, I will try to perform topic modelings with __LDA, CTM and STM__ (currently on LDA is available) on the feedback comments of a product from customers in bilingual (German and English). The overall objective is to have a glance at how the models clustering and to give support to the experts in classifying the comments.

Since there is not an existing dual-vocabulary dictionary, conducting "multilingual topic model" is not feasible in a short time span ( _"Multilingual Topic Models for Unaligned Text"_ - [https://arxiv.org/ftp/arxiv/papers/1205/1205.2657.pdf](https://arxiv.org/ftp/arxiv/papers/1205/1205.2657.pdf))

# Initialize packages

Load packages 


```r
# load packages and set options
options(stringsAsFactors = FALSE)

# install packages if not available
packages <- c("tidyverse", # set of packages for tidy
              #"remedy", # shortcuts for RMarkdown - just run 1 time
              "lubridate", #date time conversion
              "cld3","cld2","textcat", # Languages detector
              "tm","stopwords","quanteda", "topicmodels", # text analytics / topic modeling LDA
              "caTools", "RColorBrewer", # features engineering
              "ldatuning", # tuning package for LDA model
              "LDAvis" # interactive bisualization for LDA
)

if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))
}
lapply(packages, require, character.only = TRUE)
```

# Read inputs

Read the feedback as csv to a data frame. The csv is not available on the repository, but the structure is simple: it contains mainly 2 columns

+ Contents: Comments feedback (free text field)
+ Language: The language of the comment (this field is blank and will be populated in the next chunks)


```r
df <- readxl::read_xlsx("Feedback Summary Open questions.xlsx") %>%
  rename(Sentiment = 'Positive/Negative',
         Risk = 'Detailed') %>%
  select(-c(X__1,X__2)) %>%
  mutate(Sentiment = replace(Sentiment, Sentiment == 'negative', 'Negative'))
```

Simply curiously explore the dataset


```r
# Summarise by sentiment (manual)
df %>%
  group_by(Sentiment) %>%
  tally
```

```
## # A tibble: 3 x 2
##   Sentiment      n
##   <chr>      <int>
## 1 Equivalent    26
## 2 Negative     560
## 3 Positive     145
```

```r
# Summarise by category
df %>%
  group_by(Category) %>%
  tally 
```

```
## # A tibble: 12 x 2
##    Category              n
##    <chr>             <int>
##  1 Alternative Check     8
##  2 Application Entry    56
##  3 Assessment          145
##  4 Close Requisition     9
##  5 Expert Screen        80
##  6 General             116
##  7 OfferContract        24
##  8 Other                42
##  9 Pre-Screening        59
## 10 Pre-Selection        54
## 11 PRF                 110
## 12 Rejection            28
```

# Preprocessing pipeline

Remove duplicates. Detect languages using majority voting of 3 packages: cld2/3 and textcat


```r
# Extract from main df
text.df <- df 
# Remove duplicates
text.df <- text.df[!duplicated(text.df$Content),]

# Get detections
text.df <- text.df %>%
  mutate(Lang_textcat = textcat::textcat(Content),
         #Lang_franc = franc::franc(Content),
         Lang_cld2 = cld2::detect_language(Content),
         Lang_cld3 = cld3::detect_language(Content))

# Codify languages
text.df <- text.df %>%
  mutate(Lang_textcat = case_when(
    Lang_textcat == 'english' ~ 0,
    Lang_textcat == 'german' ~ 1,
    TRUE ~ 1),
    Lang_cld2 = case_when(
      Lang_cld2 == 'en' ~ 0,
      Lang_cld2 == 'de' ~ 1,
      TRUE ~ 1),
    Lang_cld3 = case_when(
      Lang_cld3 == 'en' ~ 0,
      Lang_cld3 == 'de' ~ 1,
      TRUE ~ 1))

# Get votes
text.df <- text.df %>%
  rowwise() %>%
  mutate(Lang = sum(Lang_textcat, Lang_cld2, Lang_cld3))

# Majority voting for lang detect
text.df <- text.df %>%
  rowwise() %>%
  mutate(Lang = case_when(
    Lang < 2 ~ 'en',
    Lang >=2 ~ 'de'
  )) %>%
  select (-c(Lang_textcat, Lang_cld2, Lang_cld3))

# Summarise by languages
text.df %>%
  group_by(Lang) %>%
  tally
```

```
## # A tibble: 2 x 2
##   Lang      n
##   <chr> <int>
## 1 de      533
## 2 en      169
```

Split into 2 text.en and text.de for English and German feedbacks, then proceed to build topic models with LDA on 2 datasets.

0. Remove Stopwords (languages based)
1. Remove Punctuations
2. Remove Whitespaces (with trim)
3. To lowercase

_Notice:_
In this pipeline, I haven't experimented with "Stemming", "Lemmitization" yet. 
Also, no "Part-Of-Speech" was involved. In the future if this project is continued, "POS-tagging" can help to build models that react only to nouns, adjectives, adverbs..etc. which might be useful in detecting topics of incoming documents.


```r
AposToSpace = function(x){
  x= gsub("'", ' ', x)
  x= gsub('"', ' ', x)
  return(x)
}

# Split into 2 datasets for EN and DE
text.en <- text.df %>% filter(Lang == 'en') %>% select(-Lang)
text.en <- as.data.frame(text.en)
# add sequence ID as column ID
text.en <- text.en %>% 
  mutate(id = row_number())


text.de <- text.df %>% filter(Lang == 'de') %>% select(-Lang)
text.de <- as.data.frame(text.de)
# add sequence ID as column ID
text.de <- text.de %>% 
  mutate(id = row_number())
```

# Language Corpus {.tabset .tabset-fade .tabset-pills}

## English

```r
# Cleaning
# English corpus
text.en <- text.en %>%
  rowwise() %>%
  mutate(Content = gsub("[^a-zA-Z\\s]", "" ,AposToSpace(
    tolower(
      stripWhitespace(
        removeNumbers(
          removeWords(Content, 
                      stopwords(language = 'en')))))),perl = TRUE))
```

Generating corpus and document-features matrix for English corpus


```r
#######
# English text
corp.en <- corpus(text.en$Content) # cleaned

# Document-features matrix
dfm.en <- dfm(corp.en,
              ngrams = 1L,
              stem = F)

vdfm.en <- dfm_trim(dfm.en, min_termfreq = 5, min_docfreq = 2)
# min_count = remove words used less than x
# min_docfreq = remove words used in less than x docs
```

Plot wordclouds for English corpus: with raw-term frequencies and tf-idf


```r
# English text
# Plot wordclouds (raw-term frequencies)
textplot_wordcloud(vdfm.en,  scale=c(3.5, .75), color=brewer.pal(8, "Dark2"), 
     random_order = F, rotation=0.1, max_words=250, main = "Raw Counts (en)")
```

![](1._RCM_Feedback_Topic_Modeling_v1_files/figure-html/EDA: Wordcloud for EN corpus-1.png)<!-- -->

```r
# Wordclouds (tf-idf)
textplot_wordcloud(dfm_tfidf(vdfm.en),  scale = c(3.5, .75), color = brewer.pal(8, "Dark2"), 
     random_order = F, rotation=0.1, max_words = 250, main = "TF-IDF (en)")
```

![](1._RCM_Feedback_Topic_Modeling_v1_files/figure-html/EDA: Wordcloud for EN corpus-2.png)<!-- -->

Explore dendogram to view how words are clustering:


```r
numWords <- 50

wordDfm <- dfm_tfidf(dfm_weight(vdfm.en))
wordDfm <- t(wordDfm)[1:numWords,]  # keep the top numWords words
wordDistMat <- dist(wordDfm)
wordCluster <- hclust(wordDistMat)
plot(wordCluster, xlab="", main="(EN) TF-IDF Frequency weighting (First 50 Words)")
```

![](1._RCM_Feedback_Topic_Modeling_v1_files/figure-html/EDA: Dendogram for word clustering (en)-1.png)<!-- -->

## German

```r
# German corpus
text.de <- text.de %>%
  rowwise() %>%
  mutate(Content = gsub("[^a-zA-Z\\s]", "" ,AposToSpace(
    tolower(
      stripWhitespace(
        removeNumbers(
          removeWords(Content, 
                      stopwords(language = 'de')))))),perl = TRUE))
```

Generating corpus and document-features matrix for German corpus


```r
#######
# German text
corp.de <- corpus(text.de$Content) # cleaned

# Document-features matrix
dfm.de <- dfm(corp.de,
              ngrams = 1L,
              stem = F)
vdfm.de <- dfm_trim(dfm.de, min_termfreq = 5, min_docfreq = 2)
```

Plot wordclouds for German corpus: with raw-term frequencies and tf-idf


```r
# Plot wordclouds (raw-term frequencies)
textplot_wordcloud(vdfm.de,  scale=c(3.5, .75), color=brewer.pal(8, "Dark2"), 
     random_order = F, rotation=0.1, max_words=250, main = "Raw Counts (de)")
```

![](1._RCM_Feedback_Topic_Modeling_v1_files/figure-html/EDA: Wordcloud for DE corpus-1.png)<!-- -->

```r
# Wordclouds (tf-idf)
textplot_wordcloud(dfm_tfidf(vdfm.de),  scale = c(3.5, .75), color = brewer.pal(8, "Dark2"), 
     random_order = F, rotation=0.1, max_words = 250, main = "TF-IDF (de)")
```

![](1._RCM_Feedback_Topic_Modeling_v1_files/figure-html/EDA: Wordcloud for DE corpus-2.png)<!-- -->

Explore dendogram to view how words are clustering:


```r
numWords <- 50

wordDfm <- dfm_tfidf(dfm_weight(vdfm.de))
wordDfm <- t(wordDfm)[1:numWords,]  # keep the top numWords words
wordDistMat <- dist(wordDfm)
wordCluster <- hclust(wordDistMat)
plot(wordCluster, xlab="", main="(DE) TF-IDF Frequency weighting (First 50 Words)")
```

![](1._RCM_Feedback_Topic_Modeling_v1_files/figure-html/EDA: Dendogram for word clustering (de)-1.png)<!-- -->

# Topic Modeling with LDA (Latent Dirichlet Allocation) {.tabset .tabset-fade .tabset-pills}



## English Corpus



Create _document-term matrix_ and analyze to get best __k__ (number of topics/clusters) for LDA model using package "ldatuning"


```r
dtm.en <- convert(vdfm.en, to = "topicmodels")

# Finding best k
result_k.en <- FindTopicsNumber(
  dtm.en,
  topics = seq(from = 2, to = 50, by = 2), # max and min number of topics
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"), # "Arun2010" can not run with EN corpus
  method = "Gibbs",
  control = list(seed = 1234), # set seed for reproduction
  mc.cores = 1L,
  verbose = TRUE
)
```

```
## fit models... done.
## calculate metrics:
##   Griffiths2004... done.
##   CaoJuan2009... done.
##   Arun2010... done.
##   Deveaud2014... done.
```

```r
# minimization for Arun and Cao Juan
# maximization for Griffiths and Deveaud
FindTopicsNumber_plot(result_k.en)
```

![](1._RCM_Feedback_Topic_Modeling_v1_files/figure-html/LDA: Creating dtm and find best k (EN)-1.png)<!-- -->

As can be seen from the result above for the English corpus, we can assume that the best number of topics in low-range is __8 or 16__


```r
k <- 8 # add low-range topic model first

df_lda.en <- LDA(dtm.en,
                 k,
                 method = "Gibbs",
                 control = list(
                   verbose = 100L,
                   seed = 1234, # seed for reproduction
                   iter = 500) 
                 )
```

```
## K = 8; V = 146; M = 167
## Sampling 500 iterations!
## Iteration 100 ...
## Iteration 200 ...
## Iteration 300 ...
## Iteration 400 ...
## Iteration 500 ...
## Gibbs sampling completed!
```

Use Shiny-based interactive visualization, convert model results to a json object that LDAVis requires.


```r
json <- topicmodels_json_ldavis(df_lda.en, vdfm.en, dtm.en)
new.order <- RJSONIO::fromJSON(json)$topic.order

# change open.browser = TRUE to automatically open result in browser
serVis(json, out.dir = "vis_en", open.browser = F)
```

```
## Warning in dir.create(out.dir): 'vis_en' already exists
```

```
## Loading required namespace: servr
```

<link rel="stylesheet" type="text/css" href="vis_en/lda.css">
<script src="vis_en/d3.v3.js"></script>
<script src="vis_en/ldavis.js"></script>
<iframe width="1200" height="1500" src="vis_en/index.html" frameborder="0"></iframe>


```r
# Apply topic to documents

gammaDF <- as.data.frame(df_lda.en@gamma)
names(gammaDF) <- c(1:k)

tmp <- data.frame("ID" = 1: nrow(as.data.frame(df_lda.en@documents)))

for (i in 1:nrow(as.data.frame(df_lda.en@documents))){
  tmp[i,1] <- substr(df_lda.en@documents[i],5,nchar(df_lda.en@documents[i]))
}
tmp$ID <- as.numeric(tmp$ID)

# Get topics with highest probabilities
toptopics.en <- as.data.frame(cbind(document = row.names(gammaDF), 
                                    topic = apply(gammaDF,1,function(x) names(gammaDF)[which(x==max(x))])))

toptopics.en <- cbind(tmp, toptopics.en) %>% select(-document)
colnames(toptopics.en) <- c("id","Topic")

# Join with original df for results
# Get the original df (without prepprocessing)
text.en <- text.df %>% filter(Lang == 'en')
text.en <- as.data.frame(text.en)
# add sequence ID as column ID
text.en <- text.en %>% 
  mutate(id = row_number())

text.en <- left_join(text.en,toptopics.en, by = "id")
```



## German Corpus



Create _document-term matrix_ and analyze to get best __k__ (number of topics/clusters) for LDA model using package "ldatuning"

```r
dtm.de <- convert(vdfm.de, to = "topicmodels")

# Finding best k
result_k.de <- FindTopicsNumber(
  dtm.en,
  topics = seq(from = 2, to = 50, by = 2), # max and min number of topics
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"), # "Arun2010" can not run with EN corpus
  method = "Gibbs",
  control = list(seed = 1234), # set seed for reproduction
  mc.cores = 1L,
  verbose = TRUE
)
```

```
## fit models... done.
## calculate metrics:
##   Griffiths2004... done.
##   CaoJuan2009... done.
##   Arun2010... done.
##   Deveaud2014... done.
```

```r
# minimization for Arun and Cao Juan
# maximization for Griffiths and Deveaud
FindTopicsNumber_plot(result_k.en)
```

![](1._RCM_Feedback_Topic_Modeling_v1_files/figure-html/LDA: Creating dtm and find best k (DE)-1.png)<!-- -->

As can be seen from the result above for the German corpus, we can assume that the best number of topics in low-range is __16__
The next chunk code is building LDA model and apply it on the current German corpus

```r
k <- 16 # add low-range topic model first

df_lda.de <- LDA(dtm.de,
                 k,
                 method = "Gibbs",
                 control = list(
                   verbose = 100L,
                   seed = 1234, # seed for reproduction
                   iter = 500) 
                 )
```

```
## K = 16; V = 247; M = 497
## Sampling 500 iterations!
## Iteration 100 ...
## Iteration 200 ...
## Iteration 300 ...
## Iteration 400 ...
## Iteration 500 ...
## Gibbs sampling completed!
```

Use Shiny-based interactive visualization, convert model results to a json object that LDAVis requires.


```r
json <- topicmodels_json_ldavis(df_lda.de, vdfm.de, dtm.de)
new.order <- RJSONIO::fromJSON(json)$topic.order

# change open.browser = TRUE to automatically open result in browser
serVis(json, out.dir = "vis_de", open.browser = F)
```

```
## Warning in dir.create(out.dir): 'vis_de' already exists
```

<link rel="stylesheet" type="text/css" href="vis_de/lda.css">
<script src="vis_de/d3.v3.js"></script>
<script src="vis_de/ldavis.js"></script>
<iframe width="1200" height="1500" src="vis_de/index.html" frameborder="0"></iframe>


```r
# Apply topic to documents

gammaDF <- as.data.frame(df_lda.de@gamma)
names(gammaDF) <- c(1:k)

tmp <- data.frame("ID" = 1: nrow(as.data.frame(df_lda.de@documents)))

for (i in 1:nrow(as.data.frame(df_lda.de@documents))){
  tmp[i,1] <- substr(df_lda.de@documents[i],5,nchar(df_lda.de@documents[i]))
}
tmp$ID <- as.numeric(tmp$ID)

# Get topics with highest probabilities
toptopics.de <- as.data.frame(cbind(document = row.names(gammaDF), 
                                    topic = apply(gammaDF,1,function(x) names(gammaDF)[which(x==max(x))])))

toptopics.de <- cbind(tmp, toptopics.de) %>% select(-document)
colnames(toptopics.de) <- c("id","Topic")

# Join with original df for results
# Get the original df (without prepprocessing)
text.de <- text.df %>% filter(Lang == 'de')
text.de <- as.data.frame(text.de)
# add sequence ID as column ID
text.de <- text.de %>% 
  mutate(id = row_number())

text.de <- left_join(text.de,toptopics.de, by = "id")
```
