library(stringr)
library(tidymodels)
library(tune)
library(AmesHousing)
library(xgboost)
library(yardstick)

source('mlflow_logging.R')
# ------------------------------------------------------------------------------

ames <- make_ames()

ames <- ames %>% mutate(Sale_Price = log(Sale_Price, base = 10))

set.seed(4595)
data_split <- initial_split(ames, strata = "Sale_Price", prop = 0.75)

df_train <- training(data_split)
df_test <- testing(data_split)

set.seed(2453)
rs_splits <- vfold_cv(df_train, strata = "Sale_Price", v = 4)

# ------------------------------------------------------------------------------

ames_rec <-
  recipe(Sale_Price ~ ., data = df_train) %>%
  step_YeoJohnson(Lot_Area, Gr_Liv_Area) %>%
  step_other(Neighborhood, threshold = .1)  %>%
  step_dummy(all_nominal()) %>%
  step_zv(all_predictors()) %>%
  step_ns(Longitude, deg_free = 2) %>%
  step_ns(Latitude, deg_free = 2)

xgb_spec <- boost_tree(
  trees = 250, 
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(),                     ## first three: model complexity
  sample_size = tune(), mtry = tune(),         ## randomness
  learn_rate = tune(),                         ## step size
) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

xgb_grid <- grid_random(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), df_train),
  learn_rate(),
  size = 50
)

wf_ames <- workflow() %>% add_recipe(ames_rec) %>% add_model(xgb_spec)

gs_ames <- tune_grid(
  wf_ames,
  resamples = rs_splits,
  grid = xgb_grid,
  control = control_grid(verbose = TRUE)
)

best_hyperparameters = select_best(gs_ames, metric = 'rmse')
wf_final = finalize_workflow(wf_ames, best_hyperparameters) %>% fit(data = df_train)

df_pred = wf_final %>% predict(df_test) %>% bind_cols(df_test)

metrics = yardstick::metric_set(rmse, mae, rsq)
df_metric = metrics(df_pred, Sale_Price, .pred)

# This packages the workflow up into a single bundle
library(carrier)
ames_model = carrier::crate(function(df) {
    df_pred = workflows:::predict.workflow(model, df) 
    df_pred = df_pred %>% mutate(.pred = 10^.pred)
    return(df_pred)
  }, 
  model = wf_final
)

# Point to the MLFlow tracking system, where we'll store this run. Note MLFlow is largely driven off conda,
# so we need to add it to our path.

paths = unlist(Sys.getenv('PATH') %>% str_split(':'))

conda_path = '/databricks/conda/bin'
if(!(conda_path %in% paths)) {
  Sys.setenv(PATH = paste(Sys.getenv('PATH'), conda_path, sep = ':'))  
}

library(mlflow)
install_mlflow()

mlflow_set_tracking_uri('databricks')

Sys.setenv(DATABRICKS_HOST = 'https://adb-7177174267829242.2.azuredatabricks.net') # TODO remove
Sys.setenv(DATABRICKS_TOKEN = '') # TODO remove

experiment_id = mlflow_list_experiments() %>% filter(name == "/Users/richard.boyes@centrica.com/ames_housing") %>% pull(experiment_id)

with(mlflow_start_run(experiment_id = experiment_id), {
  log_workflow_parameters(wf_final)
  log_metrics(df_metric)
  mlflow_log_model(ames_model, 'ames_housing')
})

json_test = df_test %>% select(-Sale_Price) %>% sample_n(1) %>% jsonlite::toJSON(na = "string")
model_endpoint = 'https://adb-7177174267829242.2.azuredatabricks.net/model/ames_housing/1/invocations'

httr::POST(url = model_endpoint, 
           body = df_iris_test %>% sample_n(1), 
           encode = 'json', 
           add_headers(Authorization = str_glue('Bearer {Sys.getenv("DATABRICKS_TOKEN")}')))
