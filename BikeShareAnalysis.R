library(tidymodels)
library(tidyverse)
library(vroom)

train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/train.csv")
train <- train |>
  select(-casual, -registered) |>
  mutate(weather = as_factor(weather))


test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/bikeShare/test.csv")


my_linear_model <- linear_reg() |>
  set_engine("lm") |> 
  set_mode("regression") |>
  fit(formula = count ~ . - datetime, data = train)

## Generate Predictions Using Linear Models
bike_predictions <- predict(my_linear_model,
                            new_data = test)


## kaggle submissions

kaggle_submission <- bike_predictions %>%
  bind_cols(.,test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=',')
