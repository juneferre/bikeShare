# ==============================================================================
# =============================== Data Robot  ==================================
# ==============================================================================

# clear environment /Users/juneferre/Downloads/h2o-3.46.0.7

library(tidymodels)
library(tidyverse)
library(vroom)
library(workflows)
library(recipes)
library(lubridate)
library(agua)

h2o::h2o.init()

auto_model <- auto_ml() %>%
  set_engine("h2o", max_runtime_secs= 60,
             max_models = 5) |>
  set_mode('regression')



# Reading in the data
train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/train.csv")
test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/test.csv")


# Data Cleaning
train <- train |>
  select(-casual, -registered) |> # Removing casual and registered from the train data
  mutate(count = log(count)) # transforming the count to a log count




bike_recipe <- recipe(count ~ ., data = train) |>
  # Basic datetime features
  step_mutate(
    hour   = lubridate::hour(datetime),
    wday   = lubridate::wday(datetime, week_start = 1), # Mon=1
    month  = lubridate::month(datetime),
    year   = lubridate::year(datetime),
    is_weekend = ifelse(wday >= 6, 1, 0)
  ) |>
  
  # Cyclical encodings for key cycles
  step_mutate(
    hour_sin  = sin(2 * pi * hour / 24),
    hour_cos  = cos(2 * pi * hour / 24),
    wday_sin  = sin(2 * pi * wday / 7),
    wday_cos  = cos(2 * pi * wday / 7),
    month_sin = sin(2 * pi * month / 12),
    month_cos = cos(2 * pi * month / 12)
  ) |>
  
  # Simplify weather
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_mutate(weather = factor(weather)) |>
  
  # Drop raw datetime
  step_rm(datetime, hour, wday, month) |>
  
  # Encode categoricals only
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())



prepped_recipe <- prep(bike_recipe)
baked <- bake(prepped_recipe, new_data = test)

vroom_write(x = baked, file = "./BakedTestData.csv", delim = ",")


preds <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/predictions/DataRobotResults.csv")


kaggle_submission <- preds %>%
  bind_cols(.,test) %>%
  select(datetime, count_PREDICTION) %>%
  rename(count = count_PREDICTION) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))


vroom_write(x = kaggle_submission, file = "./dataRobot1.csv", delim = ',')





