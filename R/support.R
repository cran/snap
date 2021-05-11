#' support functions for snap
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @import tensorflow
#' @import keras
#' @import dplyr
#' @import forcats
#' @import tictoc
#' @import readr
#' @import ggplot2
#' @importFrom stringr str_length
#' @importFrom stats lm median cor na.omit quantile ecdf as.formula predict
#' @importFrom utils head

globalVariables("x")

###SUPPORT
dec_points <- function(x){str_length(unlist(strsplit(as.character(x), "[.]"))[2])}

eval_metrics <- function(actual, predicted, task, positive = NULL)
{
  if(task=="regr"){metrics <- regr_metrics(actual, predicted)}
  if(task=="classif"){metrics <- classif_metrics(actual, predicted, positive)}
  if(task=="multilabel"){metrics <- multilabel_metrics(actual, predicted)}
  return(metrics)
}

classif_metrics <- function(actual, predicted, positive = NULL)
{

  actual <- unlist(actual)
  predicted <- unlist(predicted)
  if(length(actual) != length(predicted)){stop("different lengths")}
  n_length <- length(actual)
  actual_lvl <- sort(unique(actual))
  predicted_lvl <- sort(unique(predicted))
  if(!is.null(positive)){actual_lvl <- positive}

  bac <- mean(mapply(function(x) tryCatch((sum((predicted == x) & (actual == x))/sum(actual == x) + sum((predicted != x) & (actual != x))/sum(actual != x))/2, error = function(e) NA), x = actual_lvl), na.rm = TRUE)
  prc <- mean(mapply(function(x) tryCatch(sum((predicted == x) & (actual == x))/sum((predicted == x)), error = function(e) NA) , x = actual_lvl), na.rm = TRUE)
  sen <- mean(mapply(function(x) tryCatch(sum((predicted == x) & (actual == x))/sum((actual == x)), error = function(e) NA) , x = actual_lvl), na.rm = TRUE)
  csi <- mean(mapply(function(x) tryCatch(sum((predicted == x) & (actual == x))/sum((predicted == x) | (actual == x)), error = function(e) NA), x = actual_lvl), na.rm = TRUE)
  fsc <- mean(mapply(function(x) tryCatch(2 * sum((predicted == x) & (actual == x))/(sum((predicted == x) | (actual == x)) + sum((predicted == x) & (actual == x))), error = function(e) NA) , x = actual_lvl), na.rm = TRUE)
  kpp <- tryCatch(1 - (1 - mean(actual == predicted))/(1- sum(table(actual)*table(predicted))/n_length^2), error = function(e) 0)
  kdl <- tryCatch(cor(as.numeric(actual), as.numeric(predicted), use = "pairwise.complete.obs", method = "kendall"), error = function(e) 0)

  metrics <- round(c(bac = bac, prc = prc, sen = sen, csi = csi, fsc = fsc, kpp = kpp, kdl = kdl), 4)

  return(metrics)
}

###
regr_metrics <- function(actual, predicted)
{
  actual <- unlist(actual)
  predicted <- unlist(predicted)
  if(length(actual) != length(predicted)){stop("different lengths")}

  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  mdae <- median(abs(actual - predicted))
  mape <- mean(abs(actual - predicted)/actual)
  rrse <- sqrt(sum((actual - predicted)^2))/sqrt(sum((actual - mean(actual))^2))
  rae <- sum(abs(actual - predicted))/sum(abs(actual - mean(actual)))
  prsn <- cor(actual, predicted, use = "pairwise.complete.obs", method = "pearson")

  metrics <- round(c(rmse = rmse, mae = mae, mdae = mdae, mape = mape, rrse = rrse, rae = rae, prsn = prsn), 4)
  return(metrics)
}

###
multilabel_metrics <- function(actual, predicted)
{
  if(nrow(actual) != nrow(predicted)){stop("different rows")}
  if(ncol(actual) != ncol(predicted)){stop("different columns")}

  actual_lvl <- sort(unique(unlist(actual)))

  n_feats <- ncol(actual)
  n_length <- nrow(actual)

  actual <- as.matrix(actual)
  predicted <- as.matrix(predicted)

  macro_bac <- mean(mapply(function(x) tryCatch((sum((predicted == x) & (actual == x))/sum(actual == x) + sum((predicted != x) & (actual != x))/sum(actual != x))/2, error = function(e) NA), x = actual_lvl), na.rm = TRUE)
  macro_prc <- mean(mapply(function(x) tryCatch(sum((predicted == x) & (actual == x))/sum((predicted == x)), error = function(e) NA), x = actual_lvl), na.rm = TRUE)
  macro_sen <- mean(mapply(function(x) tryCatch(sum((predicted == x) & (actual == x))/sum((actual == x)), error = function(e) NA), x = actual_lvl), na.rm = TRUE)
  macro_csi <- mean(mapply(function(x) tryCatch(sum((predicted == x) & (actual == x))/sum((predicted == x) | (actual == x)), error = function(e) NA), x = actual_lvl), na.rm = TRUE)
  macro_fsc <- mean(mapply(function(x) tryCatch(2 * sum((predicted == x) & (actual == x))/(sum((predicted == x) | (actual == x)) + sum((predicted == x) & (actual == x))), error = function(e) NA), x = actual_lvl), na.rm = TRUE)
  micro_kpp <- mean(mapply(function(ft) tryCatch(1 - (1 - mean(actual[, ft] == predicted[, ft]))/(1- sum(table(actual[, ft])*table(predicted[, ft]))/n_length^2), error = function(e) NA), ft = 1:n_feats), na.rm = TRUE)
  micro_kdl <- mean(mapply(function(ft) tryCatch(cor(as.numeric(actual[, ft]), as.numeric(predicted[, ft]), use = "pairwise.complete.obs", method = "kendall"), error = function(e) NA), ft = 1:n_feats), na.rm = TRUE)

  metrics <- round(c(macro_bac = macro_bac,  macro_prc = macro_prc, macro_sen = macro_sen, macro_csi = macro_csi, macro_fsc= macro_fsc, micro_kpp = micro_kpp, micro_kdl = micro_kdl), 4)
  return(metrics)
}

###
winsorization <- function(data, q_min, q_max)
{
  range <- quantile(data, probs = c(q_min,q_max), na.rm = TRUE)
  data[data < range] <- range[1]
  data[data > range] <- range[2]
  return(data)
}


