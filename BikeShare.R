library(tidymodels)
library(tidyverse)
library(vroom)
library(workflows)
library(recipes)

# Reading in the data
train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/train.csv")
test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/test.csv")


# Data Cleaning
train <- train |>
  select(-casual, -registered) |> # Removing casual and registered from the train data
  mutate(count = log(count)) # transforming the count to a log count


# Feature engineering
# creating recipe
bike_recipe <- recipe(count ~ ., data = train) |>
  step_mutate(weather = ifelse(weather==4,3, weather)) |>
  step_mutate(weather = factor(weather)) |>
  step_time(datetime, features = c("hour", "minute")) |>
  step_mutate(season = factor(season)) |>
  step_corr(all_numeric_predictors(), threshold=0.5) |>
  step_dummy(all_nominal_predictors())|> # make dummy variables
  step+normalize(all_numeric_predictors()) # make mean 0, sd = 1 
  
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data = train)

# Define a Model 
lin_mod <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

# Combine into a Workflow and Fit
bike_workflow <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(lin_mod) |>
  fit(data = train)

# Run all steps on the test data
lin_preds <- predict(bike_workflow, new_data = test)

# Back-transform from log(count) to count
lin_preds <- lin_preds %>%
  mutate(.pred_count = exp(.pred))

head(lin_preds)


kaggle_submission <- lin_preds %>%
  bind_cols(.,test) %>%
  select(datetime, .pred_count) %>%
  rename(count = .pred_count) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x = kaggle_submission, file = "./LinearPreds2.csv", delim = ',')


