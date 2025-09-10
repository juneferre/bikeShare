library(vroom)
library(tidyverse)
library(skimr)
library(DataExplorer)
library(GGally)
library(patchwork)

train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/train.csv")
test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/test.csv")


train <- train |>
  mutate(
    season = as_factor(season),
    holiday = as_factor(holiday),
    workingday = as_factor(workingday),
    weather = as_factor(weather)
  )

skim(train)


temp <- ggplot(data=train, aes(x=temp, y=count)) + 
  geom_point() +
  geom_smooth(se=FALSE)

temp 


temp_humidity <- ggplot(data = train, aes(x = temp, y = humidity)) +
  geom_point() 

temp_humidity

weather_plot <- train |>
  count(weather) |>
  ggplot(aes(x = weather, y = n)) +
  geom_col() +
  labs(title = "Weather Counts", x = "Weather", y = "Frequency")


hist_temp <- ggplot(train, aes(x = temp)) +
  geom_histogram(bins = 30)

hist_windspeed <- ggplot(train, aes(x = windspeed)) +
  geom_histogram(bins = 30)

library(patchwork)
all <- (temp_humidity + hist_temp) / (hist_windspeed + weather_plot)
all



#ggsave(filename = "BikeShareEDA.png",plot = all, width = 8,height = 6,dpi = 300)


