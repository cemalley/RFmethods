# Load packages-----
library(foreign)
library(randomForest)
library(rfUtilities)
library(data.table)
library(dplyr)
library(reshape2)
library(ggplot2)
library(ggthemes)
library(gridExtra)
library(caret)
library(plotrix)
library(stringr)
library(Hmisc)
library(gtools)
library(ComplexHeatmap)
library(RColorBrewer)
setwd('/Users/malleyce/Documents/SAS/')

# Read in and restructure the test dataset----
data <- as.data.table(read.xport('testdata.xpt'))

# Preliminary data summary plots----
data.melted <- melt(data, id.vars=c('SUBJID', 'MCCB'))
ggplot(data.melted, aes(x=SUBJID, y=MCCB)) + geom_point()+theme_bw()
ggplot(data.melted, aes(x=SUBJID, y=value)) + geom_jitter()+theme_bw()
ggplot(data.melted, aes(x=value, y=MCCB)) + geom_jitter()+theme_bw()
summary(data)

# make discrete cutoffs for the outcome variable to be classification-- convert to 0/1 variable----
data[,MCCB_discretized:= ifelse(MCCB<9, 0, 1)]

# change SUBJIDs so that each row is a different person----
# the data should have 125 people one survey each
data[,SUBJID := c(1:125)]
data

# Print summary of responders and nonresponders. 1=responder, 0=nonresponder ----
resp <- data[data$MCCB_discretized==1,-c('MCCB')]
nonresp <- data[data$MCCB_discretized==0,-c('MCCB')]
nrow(resp)
nrow(nonresp)

# Write message saying which group is larger
if (nrow(resp) > nrow(nonresp)) {paste("More resp (",nrow(resp),") than nonresp (",nrow(nonresp),")",sep="")
}else if (nrow(nonresp) > nrow(resp)) {paste("More nonresp (",nrow(nonresp),") than resp (",nrow(resp),")",sep="")
}else {paste("Equal number of resp and nonresp (",nrow(resp),sep="")}

# Final sample size from each group. Important so that each forest has the same
# number of observations in each group.
samplesz <- min( nrow(resp),nrow(nonresp) )
paste("Sample size will be",samplesz)

# Number of forests
NForests <- 10

# Number of predictors
NVars <- length(names(data)[grep('^R', names(data))])
NVars

# Node size = 5% of number of subjects in pre-balanced file.
# Node size is how big the terminal nodes are. Should be not smaller than 5%.
nodesz <- ceiling(0.05 * 2*min(nrow(resp),nrow(nonresp)))
nodesz

# Classification, RF----
####  Prepare objects to hold results
# Data frame for error rates. One row per forest.
# Columns are out of bag (OOB) error, error for class 0, and error for class 1.
error <- data.frame(matrix(ncol=3,nrow=NForests))
names(error) <- c("oob","class0","class1")

# Data frames for variable importance ratings. One row per forest.
# DecAcc = decrease in accuracy by variable, equal to rfout$importance[,3]
# minDecAcc = Minimum decrease in accuracy, over all variables, within forest
# stdDecAcc = Decrease in accuracy, rescaled to 0-100, within forest
DecAcc <- data.frame(matrix(ncol=NForests,nrow=NVars))
names(DecAcc) <- c(paste0('Forest_',c(1:10)))
minDecAcc <- data.frame(matrix(NA, ncol=1,nrow=NForests))
names(minDecAcc) <- 'minDecAcc'
stdDecAcc <- data.frame(matrix(ncol=NVars,nrow=NForests))
names(stdDecAcc) <- names(data)[2:36]

# Data frames for mean and 2*SE rescaled importances.
# Columns are variable names. Only one row since we are aggregating over forests.
meanDecAcc <- data.frame(matrix(ncol=NVars,nrow=1))
seDecAcc <- data.frame(matrix(ncol=NVars,nrow=1))
names(meanDecAcc) <- names(data)[2:36]
names(seDecAcc) <- names(data)[2:36]
DecAcc$Feature <- names(data)[2:36]

# Do the entire random forests analysis NForests times, each one taking a different sample of resp
for (i in 1:NForests){
  # Report progress: print a note every 10th forest # changed to every 1 forest
  # if(!(i%%10)) cat(i,labels="Forest:",fill=TRUE)
  cat(i,labels="Forest:",fill=TRUE)
  # Random sample of larger file to the size of the smaller file
  if (nrow(resp) > nrow(nonresp)){
    # More resp than nonresp
    sample <- sample(1:nrow(resp),size=samplesz)
    data1 <- rbind(nonresp,resp[sample,])
  } else if ( nrow(resp) < nrow(nonresp) )
  {
    # More nonresp than resp
    sample <- sample(1:nrow(nonresp),size=samplesz)
    data1 <- rbind(nonresp[sample,],resp)
  } else
  {
    # Equal group sizes
    data1 <- rbind(nonresp,resp)
  }
  # Declare factors
  data1$MCCB_discretized <- factor(data1$MCCB_discretized)
  # RF analysis
  rfout <- randomForest(data1[,2:36],data1$MCCB_discretized,ntree=1000,nodesize=nodesz,importance=TRUE) # use default mtry
  rfout$importance[order(rfout$importance[,3], decreasing=TRUE),] # look at variable importance prior to rescaling, to determine noise
  # Save mean decrease in accuracy
  DecAcc[,i] <- rfout$importance[,3]
  # Rescale importance measures to a 0-100 scale: subtract minimum value, then divide by range, times 100.
  # For each forest, save the last record of error rates - it is cumulative over all previous trees.
  minDecAcc[i,1] <- min(rfout$importance[,3])
  stdDecAcc[i,] <- (rfout$importance[,3] - minDecAcc[i,1]) * 100/ (max(rfout$importance[,3]) - minDecAcc[i,1])
  error[i,] <- rfout$err.rate[nrow(rfout$err.rate),]
}

# Classification: Report results-----
minDecAcc
stdDecAcc
error

# Process error rates
# Mean and std error of classification error rates
MeanErrorRate <- colMeans(error)
StdErrorRate <- std.error(error, na.rm=T)
ReportErrorRate <- round(data.frame(MeanErrorRate,MeanErrorRate - 2 * StdErrorRate,MeanErrorRate + 2 * StdErrorRate)*100,digits=1)
names(ReportErrorRate) <- c("Mean Error Rate","-2SE","+2SE")
ReportErrorRate

# Sort and report mean decrease in accuracy over all forests, not standardized, easier to identify noise
RawVarImp <- rowMeans(DecAcc[,1:10])
SortVarImp <- data.frame(Sorted_Var_Imp = format(RawVarImp[order(RawVarImp, decreasing=TRUE)],scientific=FALSE))
SortVarImp

# Classification: Cross validation----
set.seed(1234)
( rf.mdl <- randomForest(data[,2:36], as.factor(data$MCCB_discretized), ntree=1000) )
( rf.cv <- rf.crossValidation(rf.mdl, data[,2:36], p=0.20, n=99, ntree=1000) )
#?rf.crossValidation()
rf.mdl
rf.cv

# Plot cross validation versus model producers accuracy
par(mfrow=c(1,2))
plot(rf.cv, type = "cv", main = "CV accuracy")
plot(rf.cv, type = "model", main = "Model accuracy")

# Plot cross validation versus model oob
par(mfrow=c(1,2))
plot(rf.cv, type = "cv", stat = "oob", main = "CV oob error")
plot(rf.cv, type = "model", stat = "oob", main = "Model oob error")

# Classification: Noise removal----
# If a feature is consistently below zero (on average
# from 10 forest runs), take out that feature.

cat('This is our forest with all features included, calculated previously.')
rfout

importances <- unlist(as.numeric(rfout$importance[,3]), use.names = F)
error <- rfout$err.rate[nrow(rfout$err.rate),]
importanceSDs <- unlist(as.numeric(rfout$importanceSD[,3]), use.names=F) 

rfout.AllFeatures <- data.table(Feature=c(1:35), Importance=importances, ImportanceSD = importanceSDs)
rfout.AllFeatures


# Plot importances for forest using all features.
ggplot(data=rfout.AllFeatures, aes(x=Feature, y=Importance)) + geom_point() +
  geom_pointrange(aes(ymin=Importance-ImportanceSD, ymax=Importance+ImportanceSD))+ theme_bw() +
  labs(title='Random Forest with all Features: Importance per Feature (Classification)')

# Remove features with importance <= 0.

features.keep <- unlist(rfout.AllFeatures[Importance> 0,c('Feature')], use.names=F)
features.removed <- unlist(rfout.AllFeatures[Importance<= 0,c('Feature')], use.names=F)
cat(paste0('Features kept: N = ', length(features.keep)))
cat(features.keep)
cat(paste0('Features Removed: N = ', length(features.removed)))
cat(features.removed)

# Filter data to features kept.

data.denoised <- subset(data, select=c(c(features.keep+1)))
data.denoised

data.denoised.outcome <- subset(data, select=c('MCCB_discretized'))

NFeatures <- length(data.denoised)
NFeatures

# Classification: Entanglement -----
# Explore relationships between the questions on the questionnaire
# Method:
# 1.	Run Random Forest (RF)
# 2.	Calculate the initial importance scores
# 3.	For each feature:
# 3a.	Drop the feature
# 3b.	Re-run RF
# 3c.	Find the change in importance
# 4.  Plot the difference between each feature before/after dropping it with the standard error bars.
# 5.  Plot a heatmap of all questions vs. each other, the change in importance

# Save initial RF importances
rf.classification <- randomForest(data.denoised,
                                  as.factor(data.denoised.outcome$MCCB_discretized), ntree=1000,
                                  nodesize=nodesz, importance=TRUE)
importances.initial <- data.table(Feature=row.names(rf.classification$importance),
                                  rf.classification$importance[,3])
names(importances.initial)[2] <- 'InitialImp'
report <- data.table(Feature=numeric(0), Used=numeric(0), Run=numeric(0),
                     Importance=numeric(0), ImportanceSD=numeric(0))
features <- row.names(rf.classification$importance)

for (i in 1:length(features)){
  Run <- i
  Feature_omitted <- features[i]
  
  data_to_use <- subset(data.denoised, select=c(names(data.denoised) %nin% Feature_omitted ))
  rfout.tmp <- randomForest(data_to_use, as.factor(data.denoised.outcome$MCCB_discretized),
                            ntree=1000, nodesize=nodesz, importance=TRUE)
  importances <- unlist(as.numeric(rfout.tmp$importance[,3]), use.names = F)
  error <- rfout.tmp$err.rate[nrow(rfout$err.rate),] # error rate for the whole forest
  importanceSDs <- unlist(as.numeric(rfout.tmp$importanceSD[,3]), use.names=F) # used to plot error bars on importance values
  
  # For debugging/watching it run:
  cat(paste0('Run: ', Run,'\nFeatures used: '))
  cat(names(data_to_use))
  cat(paste0('\nImportances: '))
  cat(importances)
  cat(paste0('\nImportance SD: '))
  cat(importanceSDs)
  cat(paste0('\n\n'))
  
  report.tmp <- data.table(Feature = names(data_to_use), Used=1, Run=Run, Importance=importances,
                           ImportanceSD=importanceSDs)
  report.tmp <-rbind(report.tmp, data.table(Feature=Feature_omitted, Used=0, Run=Run, Importance=NA,
                                            ImportanceSD=NA))
  report <- rbind(report, report.tmp)
}
report.withInitial <- merge(report, importances.initial, by=c('Feature'), all=T)
report.withInitial[,Difference := InitialImp - Importance]
report.withInitial[,Feature_num:= tstrsplit(Feature, 'R', Inf)[2]]

report.withInitial[,Feature_num := as.numeric(Feature_num)]
report.withInitial[,Importance := as.numeric(Importance)]
report.withInitial[,ImportanceSD := as.numeric(ImportanceSD)]
report.withInitial[,InitialImp := as.numeric(InitialImp)]
report.withInitial[,Difference := as.numeric(Difference)]
str(report.withInitial)

# no more sci notation
report.out <- report.withInitial
report.out[,Importance := format(Importance,scientific=F)]
report.out[,InitialImp := format(InitialImp,scientific=F)]
report.out[,Difference := format(Difference,scientific=F)]
View(report.out)

fwrite(report.out, file='RF_classification_imps.csv')

# Classification:  Entanglement: Plot change in importances for each forest----
# First forest, the importances:
plot.data.1 <- report.withInitial[Run==1,]
plot.data.1 <- plot.data.1[Feature!='R1',]
plot.data.1
ggplot(data=plot.data.1, aes(x=reorder(Feature, Feature_num), y=Importance)) + geom_point() +
  geom_pointrange(aes(ymin=Importance-ImportanceSD, ymax=Importance+ImportanceSD))+ theme_bw() +
  labs(title='Random Forest #1: Importance per Feature', x='Feature')

# Differences in importances across all forests:
# TO-DO: plot only a few per page.
plot.data <- na.omit(report.withInitial)
ggplot(data=plot.data, aes(x=reorder(Feature, Feature_num), y=Difference)) + geom_point() +
  geom_pointrange(aes(ymin=Difference-ImportanceSD, ymax=Difference+ImportanceSD))+
  facet_wrap(Run ~.)+
  theme_bw() +
  labs(title='Change in importance per feature for 35 Random Forests', x='Feature', y='Change')

# Classification: Entanglement: heatmap----
report.withInitial

htdata.CLS <- data.table()
htdata.CLS$Feature <- sort(unique(report.withInitial$Feature))
htdata.CLS
for (feature in sort(unique(report.withInitial$Feature))){
  print(feature)
  
  run.number <- report.withInitial[Feature==feature & Used==0,Run]
  this.run <- report.withInitial[Run==run.number,]
  
  initial.imp <- this.run[Feature==feature,c('InitialImp', 'Feature')]
  
  initial.imp[,1] <- 0
  
  diffs <- this.run[Feature!=feature, c('Difference', 'Feature')]
  
  names(initial.imp)[1] <- feature
  names(diffs)[1] <- feature
  
  tmp.dt <- rbind(initial.imp, diffs)
  
  htdata.CLS<- merge(htdata.CLS, tmp.dt, by='Feature', all=T)
  
}
htdata.CLS
htdata.CLS <- htdata.CLS[mixedsort(Feature)]
htdata.CLS <- subset(htdata.CLS, select=c(mixedsort(names(htdata.CLS))))

htdata.CLS.mat <- as.data.frame(htdata.CLS)
row.names(htdata.CLS.mat) <- htdata.CLS.mat$Feature
htdata.CLS.mat <- htdata.CLS.mat[-1]

#optional palette:
#cols.use <- colorRampPalette(colors=rev(RColorBrewer::brewer.pal(11,"RdBu")))(100) # reversed RdBu, creates Blue-white-red
# add col = cols.use to function below to use it

Heatmap(as.matrix(htdata.CLS.mat), row_names_side = "right",
        column_names_side = "top", show_column_names = T,
        cluster_rows = FALSE, cluster_columns = FALSE,
        heatmap_legend_param = list(legend_height = unit(8, "cm"), title='Importance'), column_title = 'Changes in importance when the feature is left out - RF Classification')

# Regression: RF----------------------------
# set up tables for collecting summary statistics from each forest.
error <- data.frame(matrix(ncol=1,nrow=NForests))
names(error) <- c("mse")

IncMSE <- data.frame(matrix(ncol=NForests,nrow=NVars))
names(IncMSE) <- c(paste0('Forest_',c(1:10)))
minIncMSE <- data.frame(matrix(NA, ncol=1,nrow=NForests))
names(minIncMSE) <- 'minIncMSE'
stdIncMSE <- data.frame(matrix(ncol=NVars,nrow=NForests))
names(stdIncMSE) <- names(data)[2:36]

meanIncMSE <- data.frame(matrix(ncol=NVars,nrow=1))
seIncMSE <- data.frame(matrix(ncol=NVars,nrow=1))
names(meanIncMSE) <- names(data)[2:36]
names(seIncMSE) <- names(data)[2:36]
IncMSE$Feature <- names(data)[2:36]


# (Regression) Do the entire random forests analysis NForests times, each one taking a different sample of resp
for (i in 1:NForests){
  # Report progress: print a note every forest
  cat(i,labels="Forest:",fill=TRUE)
  # No balancing of resp/nonresp here because response is a continuous variable MCCB.
  
  # RF analysis
  data.REG <- data[,c(2:37)]
  rfout.REG <- randomForest(data=data.REG, MCCB ~.,nodesize=nodesz,importance=TRUE, ntree=1000)
  rfout.REG$importance[order(rfout.REG$importance[,1], decreasing=TRUE),] # look at variable importance prior to rescaling, to determine noise
  # Save IncMSE, which is the importance for RF regression mode.
  IncMSE[,i] <- rfout.REG$importance[,1]
  # Rescale importance measures to a 0-100 scale: subtract minimum value, then divide by range, times 100.
  # For each forest, save the last record of error rates - it is cumulative over all previous trees.
  minIncMSE[i,1] <- min(rfout.REG$importance[,1])
  stdIncMSE[i,] <- (rfout.REG$importance[,1] - minIncMSE[i,1]) * 100/ (max(rfout.REG$importance[,1]) - minIncMSE[i,1])
  error[i,] <- rfout.REG$mse[length(rfout.REG$mse)] # mean squared error is the reported error for RF regression mode.
}


# Regression: Report results-----
minIncMSE
stdIncMSE
error

# Process error rates
# Mean and std error of regression error rates
MeanErrorRate <- colMeans(error)
StdErrorRate <- std.error(error, na.rm=T)
# ReportErrorRate <- round(data.frame(MeanErrorRate,MeanErrorRate - 2 * StdErrorRate,MeanErrorRate + 2 * StdErrorRate)*100,digits=1)
# names(ReportErrorRate) <- c("Mean MSE Rate","-2SE","+2SE")
# ReportErrorRate
# I think this is wrong for regression. it is too high

# Sort and report IncMSE over all forests, not standardized, easier to identify noise
RawVarImp <- rowMeans(IncMSE[,1:10])
SortVarImp <- data.frame(Sorted_Var_Imp = format(RawVarImp[order(RawVarImp, decreasing=TRUE)],scientific=FALSE))
SortVarImp

# Regression: Cross validation----
( rf.mdl <- randomForest(x=data[,2:35], y=data$MCCB, ntree=1000) )
( rf.cv <- rf.crossValidation(rf.mdl, data[,2:35], p=0.20, n=99, ntree=1000) )
par(mfrow=c(2,2))
plot(rf.cv)
plot(rf.cv, stat = "mse")
plot(rf.cv, stat = "var.exp")
plot(rf.cv, stat = "mae")

# Regression: Noise removal----

cat('This is our forest with all features included, calculated previously.')
rfout.REG

importances <- unlist(as.numeric(rfout.REG$importance[,1]), use.names = F)
error <- rfout.REG$mse[nrow(rfout.REG$mse)]
importanceSDs <- unlist(as.numeric(rfout.REG$importanceSD), use.names=F)

rfout.AllFeatures <- data.table(Feature=c(1:35), Importance=importances, ImportanceSD = importanceSDs)
rfout.AllFeatures


# Plot importances for forest using all features.
ggplot(data=rfout.AllFeatures, aes(x=Feature, y=Importance)) + geom_point() +
  geom_pointrange(aes(ymin=Importance-ImportanceSD, ymax=Importance+ImportanceSD))+ theme_bw() +
  labs(title='Random Forest with all Features: Importance per Feature (Regression)')

# Remove features with importance <= 0.

features.keep <- unlist(rfout.AllFeatures[Importance> 0,c('Feature')], use.names=F)
features.removed <- unlist(rfout.AllFeatures[Importance<= 0,c('Feature')], use.names=F)
cat(paste0('Features kept: N = ', length(features.keep)))
cat(features.keep)
cat(paste0('Features Removed: N = ', length(features.removed)))
cat(features.removed)

# Filter data to features kept.

data.denoised <- subset(data, select=c(c(features.keep+1), 37))
data.denoised

NFeatures <- length(data.denoised)-1
NFeatures
# Regression: Entanglement----
rf.regression <- randomForest(data=data.denoised, MCCB ~., ntree=1000,
                                  nodesize=nodesz, importance=TRUE)
importances.initial <- data.table(Feature=row.names(rf.regression$importance),
                                  rf.regression$importance[,1])
names(importances.initial)[2] <- 'InitialImp'
report <- data.table(Feature=numeric(0), Used=numeric(0), Run=numeric(0),
                     Importance=numeric(0), ImportanceSD=numeric(0))
features <- row.names(rf.regression$importance)

for (i in 1:length(features)){
  Run <- i
  Feature_omitted <- features[i]
  
  data_to_use <- subset(data.denoised, select=c(names(data.denoised) %nin% Feature_omitted))
  data_to_use$MCCB <- data.denoised$MCCB
  rfout.tmp <- randomForest(data=data_to_use, MCCB ~.,
                            ntree=1000, nodesize=nodesz, importance=TRUE)

  importances <- unlist(as.numeric(rfout.tmp$importance[,1]), use.names = F)
  error <- rfout.tmp$mse[length(rfout.tmp$mse)] # error rate for the whole forest
  importanceSDs <- unlist(as.numeric(rfout.tmp$importanceSD), use.names=F) # used to plot error bars on importance values

  # For debugging/watching it run:
  cat(paste0('Run: ', Run,'\nFeatures used: '))
  cat(names(data_to_use))
  cat(paste0('\nImportances: '))
  cat(importances)
  cat(paste0('\nImportance SD: '))
  cat(importanceSDs)
  cat(paste0('\n\n'))
  
  report.tmp <- data.table(Feature = names(data_to_use)[-length(data_to_use)], Used=1, Run=Run, Importance=importances,
                           ImportanceSD=importanceSDs)
  report.tmp <-rbind(report.tmp, data.table(Feature=Feature_omitted, Used=0, Run=Run, Importance=NA,
                                            ImportanceSD=NA))
  report <- rbind(report, report.tmp)
}
report.withInitial <- merge(report, importances.initial, by=c('Feature'), all=T)
report.withInitial[,Difference := InitialImp - Importance]
report.withInitial[,Feature_num:= tstrsplit(Feature, 'R', Inf)[2]]
report.withInitial[,Feature_num := as.numeric(Feature_num)]
report.withInitial[,Importance := as.numeric(Importance)]
report.withInitial[,ImportanceSD := as.numeric(ImportanceSD)]
report.withInitial[,InitialImp := as.numeric(InitialImp)]
report.withInitial[,Difference := as.numeric(Difference)]

fwrite(report.out, file='RF_regression_imps.csv')


# Regression: Entanglement: Plot change in importances for each forest----
# First forest, the importances:
plot.data <- report.withInitial[Run==1,]
plot.data <- na.omit(plot.data)
ggplot(data=plot.data, aes(x=reorder(Feature, Feature_num), y=Importance)) + geom_point() +
  geom_pointrange(aes(ymin=Importance-ImportanceSD, ymax=Importance+ImportanceSD))+ theme_bw() +
  labs(title='Random Forest #1: Importance per Feature (Regression)', x='Feature')

# Differences in importances across all forests:
# TO-DO: plot only a few per page.
plot.data <- na.omit(report.withInitial)
ggplot(data=plot.data, aes(x=reorder(Feature, Feature_num), y=Difference)) + geom_point() +
  geom_pointrange(aes(ymin=Difference-ImportanceSD, ymax=Difference+ImportanceSD))+
  facet_wrap(Run ~.)+
  theme_bw() +
  labs(title='Change in importance per feature for 35 Random Forests (Regression)', x='Feature', y='Change')


# Regression: Entanglement: heatmap----
report.withInitial

htdata.REG <- data.table()
htdata.REG$Feature <- sort(unique(report.withInitial$Feature))
htdata.REG
for (feature in sort(unique(report.withInitial$Feature))){
  print(feature)
  
  run.number <- report.withInitial[Feature==feature & Used==0,Run]
  this.run <- report.withInitial[Run==run.number,]
  
  initial.imp <- this.run[Feature==feature,c('InitialImp', 'Feature')]
  
  initial.imp[,1] <- 0
  
  diffs <- this.run[Feature!=feature, c('Difference', 'Feature')]
 
  names(initial.imp)[1] <- feature
  names(diffs)[1] <- feature
  
  tmp.dt <- rbind(initial.imp, diffs)

  htdata.REG<- merge(htdata.REG, tmp.dt, by='Feature', all=T)
  
}
htdata.REG
htdata.REG <- htdata.REG[mixedsort(Feature)]
htdata.REG <- subset(htdata.REG, select=c(mixedsort(names(htdata.REG))))

htdata.REG.mat <- as.data.frame(htdata.REG)
row.names(htdata.REG.mat) <- htdata.REG.mat$Feature
htdata.REG.mat <- htdata.REG.mat[-1]

#optional palette:
#cols.use <- colorRampPalette(colors=rev(RColorBrewer::brewer.pal(11,"RdBu")))(100) # reversed RdBu, creates Blue-white-red
# add col = cols.use to function below to use it

Heatmap(as.matrix(htdata.REG.mat), row_names_side = "right",
        column_names_side = "top", show_column_names = T,
        cluster_rows = FALSE, cluster_columns = FALSE,
        heatmap_legend_param = list(legend_height = unit(8, "cm"), title='Importance'), column_title = 'Changes in importance when the feature is left out - RF Regression')

# heatmap with dendrograms/clustering
# Heatmap(as.matrix(htdata.REG.mat), row_names_side = "right",
#         column_names_side = "top", show_column_names = T,
#         cluster_rows = TRUE, cluster_columns = TRUE,
#         heatmap_legend_param = list(legend_height = unit(8, "cm"), title='Importance'), column_title = 'Changes in importance when the feature is left out - RF Regression')


# Probability machine regression-----
# Use the same denoised variable set as previous regression mode, but treat the dichotomized MCCB variable as if it were continuous.

data.PM <- data.denoised
data.PM[,MCCB_discretized:= ifelse(MCCB<9, 0, 1)]
data.PM[,MCCB:=NULL]
data.PM

rfout.PM <- randomForest(data=data.PM, MCCB_discretized ~.,nodesize=nodesz,importance=TRUE, ntree=1000)
# ignore warning message.
rfout.PM

# Probability machine regression: plot prediction of being a responder----
boxplot(rfout.PM$predicted)

