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
