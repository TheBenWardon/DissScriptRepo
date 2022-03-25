SinCosData <- read.csv("E:/Python/Diss/SinCosResiduals.csv")

SinCosModel.lm <- lm(Residual ~ Label, data=SinCosData)
summary(SinCosModel.lm)





SeasonalityData <- read.csv("E:/Python/Diss/SeasonalityResiduals.csv")

SeasonalityModel.lm <- lm(Residual ~ Label, data=SeasonalityData)
summary(SeasonalityModel.lm)





AverageData <- read.csv("E:/Python/Diss/AverageResiduals.csv")

AverageModel.lm <- lm(Residual ~ Label, data=AverageData)
summary(AverageModel.lm)
