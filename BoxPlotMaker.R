library(ggplot2)

LeukDataTPU <- data.frame(InferenceTime = read.csv("E:/Python/Diss/InferenceCSVs/LeukInference.csv")[,1], Hardware = c("TPU"))
LeukDataCPU <- data.frame(InferenceTime = read.csv("E:/Python/Diss/InferenceCSVs/LeukCPUInference.csv")[,1], Hardware = c("CPU"))

LeukData <- rbind(LeukDataCPU, LeukDataTPU)
LeukData$InferenceTime <- LeukData$InferenceTime * 1000.0

# LeukGG <- ggplot(LeukData, aes(x =Hardware, y=InferenceTime)) + 
#   geom_boxplot() +
#   theme_bw() +
#   theme(panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         panel.background = element_blank(),
#         axis.line = element_line(colour = "black")
#         #axis.text.x = element_blank(),
#         #axis.ticks.x = element_blank()
#         #axis.title.x = element_blank()
#         ) +
#   labs(y = "Inference Time (ms)")
# 
# print(LeukGG)






CovidDataTPU <- data.frame(InferenceTime = read.csv("E:/Python/Diss/InferenceCSVs/CovidInference.csv")[,1], Hardware = c("TPU"))
CovidDataCPU <- data.frame(InferenceTime = read.csv("E:/Python/Diss/InferenceCSVs/CovidCPUInference.csv")[,1], Hardware = c("CPU"))

CovidData <- rbind(CovidDataCPU, CovidDataTPU)
CovidData$InferenceTime <- CovidData$InferenceTime * 1000.0
# 
# CovidGG <- ggplot(CovidData, aes(x =Hardware, y=InferenceTime)) + 
#   geom_boxplot() +
#   theme_bw() +
#   theme(panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         panel.background = element_blank(),
#         axis.line = element_line(colour = "black")
#         #axis.text.x = element_blank(),
#         #axis.ticks.x = element_blank()
#         #axis.title.x = element_blank()
#   ) +
#   labs(y = "Inference Time (ms)")
# 
# print(CovidGG)







ECGDataTPU <- data.frame(InferenceTime = read.csv("E:/Python/Diss/InferenceCSVs/ECGInference.csv")[,1], Hardware = c("TPU"))
ECGDataCPU <- data.frame(InferenceTime = read.csv("E:/Python/Diss/InferenceCSVs/ECGCPUInference.csv")[,1], Hardware = c("CPU"))

ECGData <- rbind(ECGDataCPU, ECGDataTPU)
ECGData$InferenceTime <- ECGData$InferenceTime * 1000.0

# ECGGG <- ggplot(ECGData, aes(x =Hardware, y=InferenceTime)) + 
#   geom_boxplot() +
#   theme_bw() +
#   theme(panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         panel.background = element_blank(),
#         axis.line = element_line(colour = "black")
#         #axis.text.x = element_blank(),
#         #axis.ticks.x = element_blank()
#         #axis.title.x = element_blank()
#   ) +
#   labs(y = "Inference Time (ms)")
# 
# print(ECGGG)

