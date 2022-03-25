library(ggplot2)

CreateTrainValidationDataFrame <- function(TrainDirectory, ValidationDirectory){
  #Train <- read.csv(TrainDirectory)
  #Train <- Train$Value
  #Train <- data.frame(Train)
  #colnames(Train) <- c("TrainAccuracy")
  
  #Validation <- read.csv(ValidationDirectory)
  #Validation <- Validation$Value
  #Validation <- data.frame(Validation)
  #colnames(Validation) <- c("ValidationAccuracy")
  
  Train <- read.csv(TrainDirectory)
  Train <- Train$Value
  Validation <- read.csv(ValidationDirectory)
  Validation <- Validation$Value
  
  TrainValidation <- data.frame(TrainAccuracy = Train, ValidationAccuracy = Validation)
  
  TrainValidation$Index <- 1:nrow(TrainValidation)
  
  return(TrainValidation)
}

TrainValidationDataFrameFromSingle <- function(LogFolderBase){
  TrainDir <- paste(LogFolderBase, "_train.csv", sep="")
  ValDir <- paste(LogFolderBase, "_validation.csv", sep="")
  
  TrainValDF <- CreateTrainValidationDataFrame(TrainDir, ValDir)
  
  return(TrainValDF)
}

PlotFromDir <- function(Dir){
  TrainValDF <- TrainValidationDataFrameFromSingle(Dir)
  
  ThePlot <- ggplot(data=TrainValDF, aes(Index)) +
    geom_line(aes(y = TrainAccuracy, color = "TrainAccuracy")) +
    geom_line(aes(y = ValidationAccuracy, color = "ValidationAccuracy")) +
    scale_colour_manual(name="Curve", values = c("TrainAccuracy" = "blue", "ValidationAccuracy" = "orange"))+
    ylim(0.0, 1.0) +
    theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour = "black")
    ) +
    labs(y = "Accuracy", x = "Epochs") +
    theme(legend.position="none")
  
  ggsave(paste(Dir, ".png", sep=""), dpi=300, width=4.5, height=3)
  
  print(ThePlot)
  
  return(TrainValDF)
}


# PlotFromDir("E:/Python/Diss/AccuracyCSVs/ECGClassification_MLP-1647618930")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/ECGClassification_MLP_Full-1647275876")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/ECGClassification_BiRNN_GRU-1647625927")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/ECGClassification_BiRNN_LSTM-1647619436")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/ECGClassification_RNN_GRU-1647698616")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/ECGClassification_RNN_LSTM-1647701132")
# 
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/LeukemiaBinaryClassification_HomeBrewResNet_FOLD0-1646680995")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/LeukemiaBinaryClassification_HomeBrewResNet_FOLD1-1646690126")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/LeukemiaBinaryClassification_HomeBrewResNet_FOLD2-1646700379")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/LeukemiaBinaryClassification_VGG16_FOLD0-1646601298")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/LeukemiaBinaryClassification_VGG16_FOLD1-1646605461")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/LeukemiaBinaryClassification_VGG16_FOLD2-1646608677")
# 
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/CovidClassification_VGG16_MFCC-1647440796")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/CovidClassification_MFCC_JustCOVID-1647966383")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/CovidClassification_Spectrogram_JustCOVID-1647880700")
# PlotFromDir("E:/Python/Diss/AccuracyCSVs/CovidClassification_Spectrogram-1647949819")
#
PlotFromDir("E:/Python/Diss/AccuracyCSVs/CovidClassification_Multi_MFCC-1648076484")
PlotFromDir("E:/Python/Diss/AccuracyCSVs/CovidClassification_Multi_Spectrogram-1648114920")


