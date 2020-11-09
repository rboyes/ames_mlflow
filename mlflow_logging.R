library(tidymodels)

log_workflow_parameters <- function(workflow) {
  # Would help to have a check here: has this workflow been finalised?
  # It may be sufficient to check that the arg quosures carry no environments.
  spec <- workflows::pull_workflow_spec(workflow)
  parameter_names <- names(spec$args)
  parameter_values <- lapply(spec$args, rlang::get_expr)
  for (i in seq_along(spec$args)) {
    parameter_name <- parameter_names[[i]]
    parameter_value <- parameter_values[[i]]
    if (!is.null(parameter_value)) {
      mlflow_log_param(parameter_name, parameter_value)
    }
  }
  return(workflow)
}

log_metrics <- function(metrics, estimator = "standard") {
  metrics %>% filter(.estimator == estimator) %>% pmap(
    function(.metric, .estimator, .estimate) {
      mlflow_log_metric(.metric, .estimate)  
    }
  )
  return(metrics)
}