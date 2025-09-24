# ==============================================================================
# ============================== Regression Tree ===============================
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

my_mod <- decision_tree(tree_depth = tune(),
                       cost_complexity = tune(),
                       min_n = tune()) |>
  set_engine("rpart") |>
  set_mode("regression")


# Feature engineering
bike_recipe <- recipe(count ~ ., data = train) |>
  step_mutate(
    hour = lubridate::hour(datetime),
    day = lubridate::wday(datetime, week_start = 1), # 1=Monday ... 7=Sunday
    month = lubridate::month(datetime)
  ) |>
  step_mutate(
    hour_sin = sin(2 * pi * hour / 24),
    hour_cos = cos(2 * pi * hour / 24),
    day_sin  = sin(2 * pi * day / 7),
    day_cos  = cos(2 * pi * day / 7),
    month_sin = sin(2 * pi * month / 12),
    month_cos = cos(2 * pi * month / 12)
  ) |>
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_mutate(weather = factor(weather)) |>
  step_rm(datetime, hour, day, month) |>   # drop raw versions
  step_corr(all_numeric_predictors(), threshold = 0.5) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())


prepped_recipe <- prep(bike_recipe)
baked <- bake(prepped_recipe, new_data = train)



# set workflow
tree_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(my_mod)

# Grid of values to tune over
grid_of_tuning_params <- grid_regular(tree_depth(),
                                      cost_complexity(),
                                      min_n(),
                                      levels = 3)

# Split data for CV
folds <- vfold_cv(train, v = 5, repeats = 1)

# Run the Cross validation (CV)

CV_results <- tree_wf |>
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse,mae))


# Find best Tuning Parameters
bestTune <- CV_results |>
  select_best(metric="rmse")



final_wf <- 
  tree_wf |>
  finalize_workflow(bestTune) |>
  fit(data = train)


# Back-transform from log(count) to count
# Generate predictions on the test data
preds <- predict(final_wf, new_data = test) %>%
  mutate(.pred_count = exp(.pred))


kaggle_submission <- preds %>%
  bind_cols(.,test) %>%
  select(datetime, .pred_count) %>%
  rename(count = .pred_count) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))


vroom_write(x = kaggle_submission, file = "./reg_tree1.csv", delim = ',')





