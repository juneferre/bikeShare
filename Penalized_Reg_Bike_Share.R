# ==============================================================================
# ============================ Penalized Regression ============================
# ==============================================================================

# clear environment 

library(tidymodels)
library(tidyverse)
library(vroom)
library(workflows)
library(recipes)
library(lubridate)

# Reading in the data
train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/train.csv")
test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/test.csv")


# Data Cleaning
train <- train |>
  select(-casual, -registered) |> # Removing casual and registered from the train data
  mutate(count = log(count)) # transforming the count to a log count


# Feature engineering

bike_recipe <- recipe(count ~ ., data = train) |>
  step_mutate(weather = ifelse(weather==4,3, weather)) |>
  step_mutate(weather = factor(weather)) |>
  step_mutate(hour_of_week = lubridate::wday(datetime, week_start = 1) * 24 +
                lubridate::hour(datetime)) |>
  step_rm(datetime) |>  # remove original datetime column
  step_corr(all_numeric_predictors(), threshold = 0.5) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |> # remove zero-variance predictors
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data = train)

# penalized regression model
preg_model <- linear_reg(penalty=.01, mixture=.001) |>
  set_engine("glmnet")
preg_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(preg_model) |>
  fit(data = train)

preds <- predict(preg_wf, new_data = test)

# Back-transform from log(count) to count
preds <- preds %>%
  mutate(.pred_count = exp(.pred))

head(preds)


kaggle_submission <- preds %>%
  bind_cols(.,test) %>%
  select(datetime, .pred_count) %>%
  rename(count = .pred_count) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))


vroom_write(x = kaggle_submission, file = "./penalty_reg5.csv", delim = ',')


