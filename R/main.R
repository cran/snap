#' snap
#'
#' @param data A data frame including all the features and targets.
#' @param target String. Single label for target feature when task is "regr" or "classif". String vector with multiple labels for target features when task is "multilabel".
#' @param task String. Inferred by data type of target feature(s). Available options are: "regr", "classif", "multilabel". Default: NULL.
#' @param positive String. Positive class label (only for classification task). Default: NULL.
#' @param skip_shortcut Logical. Option to add a skip shortcut to improve network performance in case of many layers. Default: FALSE.
#' @param embedding String. Available options are: "none", "global" (when identical values for different features hold different meanings), "sequence" (when identical values for different features hold the same meaning). Default: NULL.
#' @param embedding_size Integer. Output dimension for the embedding layer. Default: 10.
#' @param folds Positive integer. Number of folds for repeated cross-validation. Default: 3.
#' @param reps Positive integer. Number of repetitions for repeated cross-validation. Default: 1.
#' @param holdout Positive numeric. Percentage of cases for holdout validation. Default: 0.3.
#' @param layers Positive integer. Number of layers for the neural net. Default: 1.
#' @param activations String. String vector with the activation functions for each layer (for example, a neural net with 3 layers may have activations = c("relu", "gelu", "tanh")). Besides standard Tensorflow/Keras activations, you can also choose: "swish", "mish", "gelu", "bent". Default: "relu".
#' @param regularization_L1 Positive numeric. Value for L1 regularization of the loss function. Default: 0.
#' @param regularization_L2 Positive numeric. Value for L2 regularization of the loss function. Default: 0.
#' @param nodes Positive integer. Integer vector with the nodes for each layer (for example, a neural net with 3 layers may have nodes = c(32, 64, 16)). Default: 32.
#' @param dropout Positive numeric. Value for the dropout parameter for each layer (for example, a neural net with 3 layers may have dropout = c(0, 0.5, 0.3)). Default: 0.
#' @param span Positive numeric. Percentage of epoch for the patience parameter. Default: 0.2.
#' @param min_delta Positive numeric. Minimum improvement on metric to trigger the early stop. Default: 0.
#' @param batch_size Positive integer. Maximum batch size for training. Default: 32.
#' @param epochs Positive integer. Maximum number of forward and backward propagations. Default: 50.
#' @param imp_thresh Positive numeric. Importance threshold (in percentiles) above which the features are included in the model (using ReliefFbestK metric by CORElearn). Default: 0 (all features included).
#' @param anom_thresh Positive numeric. Anomaly threshold (in percentiles) above which the instances are excluded by the model (using lof by dbscan). Default: 1 (all instances included).
#' @param output_activation String. Default: NULL. If not specified otherwise, it will be "Linear" for regression task, "Softmax" for classification task, "Sigmoid" for multilabel task.
#' @param optimizer String. Standard Tensorflow/Keras Optimization methods are available. Default: "Adam".
#' @param loss Default: NULL. If not specified otherwise, it will be "mean_absolute_error" for regression task, "categorical_crossentropy" for classification task, "binary_crossentropy" for multilabel task.
#' @param metrics Default: NULL. If not specified otherwise, it will be "mean_absolute_error" for regression task, "categorical_crossentropy" for classification task, "binary_crossentropy" for multilabel task.
#' @param winsor Logical. Set to TRUE in case you want to perform Winsorization on regression tasks. Default: FALSE.
#' @param q_min Positive numeric. Minimum quantile threshold for Winsorization. Default: 0.01.
#' @param q_max Positive numeric. Maximum quantile threshold for Winsorization. Default: 0.99.
#' @param normalization Logical. After each layer it performs a batch normalization. Default: TRUE.
#' @param seed Positive integer. Seed value to control random processes. Default: 42.
#' @param verbose Positive integer. Set the level of information from Keras. Default: 0.
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @return This function returns a list including:
#' \itemize{
#' \item task: kind of task solved
#' \item configuration: main hyper-parameters describing the neural net (layers, activations, regularization_L1, regularization_L2, nodes, dropout)
#' \item model: Keras standard model description
#' \item pred_fun: function to use on the same data scheme to predict new values
#' \item plot: Keras standard history plot
#' \item testing_frame: testing set with the related predictions, including
#' \item trials: statistics for each trial during the repeated cross-validation (train set and validation set):
#'  \itemize{
#' \item task "classif": balanced accuracy (bac), precision (prc), sensitivity (sen), critical success index (csi), FALSE-score (fsc), Kappa (kpp), Kendall (kdl)
#' \item task "regr": root mean square error(rmse), mean absolute error (mae), median absolute error (mdae), relative root square error (rrse), relative absolute error (rae), Pearson (prsn)
#' \item task "multilabel": macro bac, macro prc, macro sensitivity, macro sen, macro csi, macro fsc, micro kpp, micro kdl
#' }
#' \item metrics: summary statistics as above for training, validation (both averaged over trials) and testing
#' \item selected_feat: labels of features included within the model
#' \item selected_inst: index of instances included within the model
#' \item time_log
#' }
#'
#'
#' @export
#'
#' @import keras
#' @import tensorflow
#' @import dplyr
#' @import purrr
#' @import forcats
#' @import tictoc
#' @import readr
#' @import ggplot2
#' @importFrom CORElearn attrEval
#' @importFrom dbscan lof
#' @importFrom reticulate py_set_seed
#'
#'@examples
#'\dontrun{
#'snap(friedman3, target="y")
#'
#'snap(threenorm, target="classes", imp_thresh = 0.3, anom_thresh = 0.95)
#'
#'snap(threenorm, "classes", layers = 2, activations = c("gelu", "swish"), nodes = c(32, 64))
#'}
#'
snap <-function(data, target, task=NULL, positive=NULL,
                skip_shortcut = FALSE, embedding = "none", embedding_size = 10, folds=3, reps=1, holdout=0.3,
                layers = 1, activations = "relu", regularization_L1 = 0, regularization_L2 = 0, nodes = 32, dropout = 0,
                span=0.2, min_delta=0, batch_size=32, epochs=50, imp_thresh  = 0, anom_thresh = 1,
                output_activation=NULL, optimizer = "Adam", loss = NULL, metrics = NULL,
                winsor = FALSE, q_min = 0.01, q_max = 0.99, normalization = TRUE,
                seed = 42, verbose = 0)

{
  config <- tf$compat$v1$ConfigProto(gpu_options = list(allow_growth = TRUE))
  sess <- tf$compat$v1$Session(config = config)

  ###SUPPORT
  pred_fun <- function(new)
  {

    new <- new[, selected_feat, drop=FALSE]
    if(is.data.frame(new)){new <- data.matrix(new)}
    if(global_embedding==TRUE){new <- new + matrix(cum_max_range, nrow=nrow(new), ncol=length(cum_max_range), byrow = TRUE)}

    if(task=="regr")
    {
      prediction <- as.data.frame(predict(model, new, batch_size = batch_size))
      prediction <- as.data.frame(map2(prediction, dpoints, ~ round(.x, .y)))
      colnames(prediction) <- paste0("predicted_", target_names)
    }

    if(task == "classif")
    {
      prediction_prob<-predict(model, new, batch_size=batch_size)
      colnames(prediction_prob) <- paste0("prob_class_", class_names)
      prediction_class <- factor(class_names[apply(prediction_prob, 1, which.max)], levels = class_names)
      prediction <- data.frame(prediction_prob, prediction_class)
    }

    if(task == "multilabel")
    {
      prediction <- predict(model, new, batch_size=batch_size)
      prediction <- as.data.frame(mapply(function(lvl, s, e) factor(lvl[apply(prediction[,s:e,drop=FALSE], 1, which.max)], levels = lvl), lvl=class_names, s=start_index, e=end_index))
      colnames(prediction) <- paste0("predicted_", target_names)
    }

    return(prediction)
  }

  tic.clearlog()
  tic("snap")

  ###CHECK FOR DATAFRAME
  if(!is.data.frame(data)){data <- as.data.frame(data)}
  x_data <- data[, setdiff(colnames(data), target), drop = FALSE]
  y_data <- data[, target, drop = FALSE]
  if(is.null(task) & (is.factor(unlist(y_data)) | is.character(unlist(y_data)) | is.logical(unlist(y_data))) & length(target) > 1) {task <- "multilabel"}
  if(is.null(task) & (is.factor(unlist(y_data)) | is.character(unlist(y_data)) | is.logical(unlist(y_data)))) {task <- "classif"}
  if(is.null(task) & is.numeric(unlist(y_data))){task <- "regr"}
  target_names <- colnames(y_data)
  x_orig <- x_data
  y_orig <- y_data

  ####Y_DATA PREPARATION FOR CLASSIF TASK
  if(task=="regr"){dpoints <- map_dbl(y_data[1,], ~ dec_points(.x))}

  if(task=="classif")
  {
    y_data <- factor(as.character(unlist(y_data)))
    if(!is.null(positive) & length(levels(y_data))==2){y_data <- fct_relevel(y_data, positive)}
    y_orig <- as.data.frame(y_data)
    class_names <- levels(y_data)
    class_num <- length(class_names)
    if(class_num < 2){stop("need at least two levels for classification")}
    y_data <- as.data.frame(to_categorical(as.numeric(y_data)-1, num_classes = class_num))
    colnames(y_data) <- paste0("level_", class_names)
  }


  if(task=="multilabel")
  {
    #class_names <- replicate(ncol(y_data), c(0, 1), simplify = F)
    #class_num <- map_dbl(class_names, ~length(.x))
    #start_index <- head(c(1, cumsum(class_num)+1),-1)
    #end_index <- cumsum(class_num)
    y_data <- as.data.frame(lapply(y_data, as.character))

    #if(!all(unlist(y_data)%in% c(0, 1)))
    #{
      y_data <- as.data.frame(lapply(y_data, factor))
      y_orig <- y_data
      class_names <- lapply(y_data, levels)
      class_num <- map_dbl(class_names, ~length(.x))
      start_index <- head(c(1, cumsum(class_num)+1),-1)
      end_index <- cumsum(class_num)
      if(any(class_num < 2)){stop("need at least two levels for each label")}
      y_names <- unlist(map2(colnames(y_data), class_names, ~paste0(.x,"_level_", .y)))
      y_data <- as.data.frame(map2(y_data, class_num, ~to_categorical(as.numeric(.x)-1, num_classes = .y)))
      colnames(y_data) <- y_names
    #}
  }

  if(task=="regr" & winsor == TRUE)
  {
    y_data <- apply(y_data, 2, winsorization, q_min = q_min, q_max = q_max)
  }

  ###EMBEDDING
  if(embedding == "none"){global_embedding <- FALSE; sequence_embedding <- FALSE}
  if(embedding == "global"){global_embedding <- TRUE; sequence_embedding <- FALSE}
  if(embedding == "sequence"){sequence_embedding <- TRUE; global_embedding <- FALSE}

  ###SELECTION
  if(task=="multilabel" & imp_thresh > 0){imp_thresh <- 0; message("Feature selection not available for multilabel task. Setting imp_thresh to zero.")}
  if(imp_thresh > 0)
  {
    target <- paste0(colnames(y_orig), collapse = " + ")
    importance_scores <- suppressWarnings(attrEval(as.formula(paste0(target, "~.")), cbind(y_orig, x_orig), estimator=ifelse(task == "regr", "RReliefFbestK", "ReliefFbestK")))
    importance_scores <- ecdf(importance_scores)(importance_scores)
    selected_feat <- colnames(x_orig)[importance_scores >= imp_thresh]
  }
  else {selected_feat <- colnames(x_orig)}

  if(anom_thresh < 1)
  {
    anomaly_scores <- apply(as.data.frame(map(10:max(30, round(sqrt(nrow(x_data)))), ~ lof(data.matrix(x_data), k = .x))), 1, max)###K TRESH, SOME REFERENCE ON STACKOV. & RESEARCH PAPERS
    anomaly_scores <- ecdf(anomaly_scores)(anomaly_scores)
    selected_inst <- c(1:nrow(x_data))[anomaly_scores <= anom_thresh]
  }
  else {selected_inst <- 1:nrow(x_data)}

  x_selected <- x_data[selected_inst, selected_feat, drop = FALSE]
  y_selected <- y_data[selected_inst, , drop = FALSE]


  ###SWITCH FROM DATAFRAME TO ARRAY
  x_array <- data.matrix(x_selected)
  y_array <- data.matrix(y_selected)

  x_rows<-dim(x_array)[1]
  x_cols<-dim(x_array)[2]
  y_rows<-dim(y_array)[1]
  y_cols<-dim(y_array)[2]

  set.seed(seed)
  test_index<-sample(x_rows, ceiling(holdout*x_rows))
  train_index<-setdiff(c(1:x_rows), test_index)

  x_train <- x_array[train_index,, drop=FALSE]
  y_train <- y_array[train_index,, drop=FALSE]
  x_test <- x_array[test_index,, drop=FALSE]
  y_test <- y_array[test_index,, drop=FALSE]


  ###DESIGN OF A SINGLE  NETWORK

  if(length(activations)<layers){activations <- replicate(layers, activations[1])}
  if(length(regularization_L1)<layers){regularization_L1 <- replicate(layers, regularization_L1[1])}
  if(length(regularization_L2)<layers){regularization_L2 <- replicate(layers, regularization_L2[1])}
  if(length(nodes)<layers){nodes <- replicate(layers, nodes[1])}
  if(length(dropout)<layers){dropout <- replicate(layers, dropout[1])}

  configuration<-data.frame(layers = NA, activations = NA, regularization_L1 = NA, regularization_L2 = NA, nodes = NA, dropout = NA)
  configuration$layers <- layers
  configuration$activations <- list(activations)
  configuration$regularization_L1 <- list(regularization_L1)
  configuration$regularization_L2 <- list(regularization_L2)
  configuration$nodes <- list(nodes)
  configuration$dropout <- list(dropout)

  ###CREATION OF KERAS NEURAL NET MODELS
  input_tensor <- layer_input(shape= dim(x_array)[-1])
  interim <- input_tensor

  ####EMBEDDING
  if(sequence_embedding == TRUE)
  {
    vocab_size <- max(x_array, na.rm = TRUE) + 1
    interim <- layer_embedding(object = interim, input_dim = vocab_size, output_dim = embedding_size)
    interim <- layer_flatten(object = interim)
  }


  if(global_embedding == TRUE)
  {
    max_range <- apply(x_array, 2, max, na.rm=TRUE)
    cum_max_range <- c(0, head(cumsum(max_range), -1))
    x_array <- x_array + matrix(cum_max_range, nrow=nrow(x_array), ncol=length(cum_max_range), byrow = TRUE)

    vocab_size <- max(x_array, na.rm = TRUE) + 1
    interim <- layer_embedding(object = interim, input_dim = vocab_size, output_dim = embedding_size)
    interim <- layer_flatten(object = interim)
  }

  for(l in 1:configuration$layers)
  {
    interim <- layer_dense(object = interim, units = unlist(configuration$nodes)[l],
                           kernel_regularizer = regularizer_l1_l2(l1=unlist(configuration$regularization_L1)[l],
                                                                  l2=unlist(configuration$regularization_L2)[l]))

    swish <- function(x, beta = 1){x * k_sigmoid(beta * x)}
    mish <- function(x){x * k_tanh(k_softplus(x))}
    gelu <- function(x){0.5 * x * (1 + k_tanh(sqrt(2/pi) * (x + 0.044715 * x ^ 3)))}
    bent <- function(x){(sqrt(x^2 + 1) - 1) / 2 + x}

    checklist<-c("elu", "relu", "selu", "leaky_relu", "parametric_relu", "thresholded_relu", "softmax", "swish", "mish", "gelu", "bent")
    if(unlist(configuration$activations)[l]=="elu") {interim<- layer_activation_elu(object=interim)}
    if(unlist(configuration$activations)[l]=="relu") {interim<- layer_activation_relu(object=interim)}
    if(unlist(configuration$activations)[l]=="leaky_relu") {interim<- layer_activation_leaky_relu(object=interim)}
    if(unlist(configuration$activations)[l]=="parametric_relu") {interim<- layer_activation_parametric_relu(object=interim)}
    if(unlist(configuration$activations)[l]=="thresholded_relu") {interim<- layer_activation_thresholded_relu(object=interim)}
    if(unlist(configuration$activations)[l]=="selu") {interim<- layer_activation_selu(object=interim)}
    if(unlist(configuration$activations)[l]=="softmax") {interim<- layer_activation_softmax(object=interim)}
    if(unlist(configuration$activations)[l]=="swish") {interim<- layer_activation(object=interim, activation = swish)}
    if(unlist(configuration$activations)[l]=="mish") {interim<- layer_activation(object=interim, activation = mish)}
    if(unlist(configuration$activations)[l]=="gelu") {interim<- layer_activation(object=interim, activation = gelu)}
    if(unlist(configuration$activations)[l]=="bent") {interim<- layer_activation(object=interim, activation = bent)}
    if(!(unlist(configuration$activations)[l] %in% checklist)){interim<- layer_activation(object=interim, activation = unlist(configuration$activations)[l])}

    interim<-layer_dropout(object=interim, rate=unlist(configuration$dropout)[l])
    if(normalization==TRUE){interim<-layer_batch_normalization(object=interim)}
  }

  if(skip_shortcut==TRUE)
  {
    reshaped <- layer_dense(object=interim, units = dim(x_array)[-1])
    interim <- layer_add(list(reshaped, input_tensor))
  }

  if(is.null(output_activation) & task=="regr"){output_activation="linear"} ###DEFAULT VALUE FOR REGR PROBLEMS
  if(is.null(output_activation) & task=="classif"){output_activation="softmax"} ###DEFAULT VALUE FOR CLASSIF PROBLEMS
  if(is.null(output_activation) & task=="multilabel"){output_activation="sigmoid"} ###DEFAULT VALUE FOR CLASSIF PROBLEMS

  if(task=="regr"|task=="classif"|task=="multilabel"){output_tensor <- layer_dense(object=interim, activation= output_activation, units = dim(y_array)[-1])}

  model <- keras_model(inputs = input_tensor, outputs = output_tensor)

  ###DEFAULT VALUES FOR MODEL COMPILE
  if(is.null(loss) & task=="regr"){loss="mean_absolute_error"}
  if(is.null(loss) & task=="classif"){loss="categorical_crossentropy"}
  if(is.null(loss) & task=="multilabel"){loss="binary_crossentropy"}
  if(is.null(metrics) & task=="regr"){metrics="mean_absolute_error"}
  if(is.null(metrics) & task=="classif"){metrics="categorical_accuracy"}
  if(is.null(metrics) & task=="multilabel"){metrics="binary_accuracy"}

  compile(object=model, loss = loss, optimizer = optimizer, metrics = metrics)

  rep_train_metrics <- vector("list", length = reps)
  rep_val_metrics <- vector("list", length = reps)

  for(r in 1:reps) ### CYCLES FOR CROSSVALIDATION WITH REPETITION
  {
    set.seed(seed+r)
    fold_index <- sample(folds, nrow(x_train), replace=TRUE)

    fold_train_metrics <- vector("list", length = folds)
    fold_val_metrics <- vector("list", length = folds)

    for(k in 1:folds)
    {
      x_fold_k <- x_train[which(fold_index==k), , drop = FALSE]
      y_fold_k <- y_train[which(fold_index==k), , drop = FALSE]
      x_fold_non_k <- x_train[which(fold_index!=k), , drop = FALSE]
      y_fold_non_k <- y_train[which(fold_index!=k), , drop = FALSE]

      ###MODEL FIT
      history <- model %>% fit(x_fold_non_k, y_fold_non_k, epochs = epochs, batch_size=batch_size, verbose = verbose,
                               validation_data = list(x_fold_k, y_fold_k), callbacks = list(callback_early_stopping(monitor="val_loss", min_delta=min_delta, patience=floor(epochs*span), restore_best_weights=TRUE)))

      y_orig_fold_k <- y_orig[train_index,,drop=FALSE][fold_index==k,,drop=FALSE]###REMAKING YFOLDS FOR CUSTOM METRIC EVALUATION
      y_orig_fold_non_k <- y_orig[train_index,,drop=FALSE][fold_index!=k,,drop=FALSE]###REMAKING YFOLDS FOR CUSTOM METRIC EVALUATION

      train_prediction <- pred_fun(x_fold_non_k)
      if(task == "classif"){train_predicted <- train_prediction[, "prediction_class", drop = FALSE]; train_probs <- train_prediction[, 1:class_num, drop = FALSE]} else {train_predicted <- train_prediction}
      fold_train_metrics[[k]] <- suppressWarnings(eval_metrics(y_orig_fold_non_k, train_predicted, task, positive))

      val_prediction <- pred_fun(x_fold_k)
      if(task == "classif"){val_predicted <- val_prediction[, "prediction_class", drop = FALSE]; val_probs <- val_prediction[, 1:class_num, drop = FALSE]} else {val_predicted <- val_prediction}
      fold_val_metrics[[k]] <- suppressWarnings(eval_metrics(y_orig_fold_k, val_predicted, task, positive))
    }

    rep_train_metrics[[r]] <- fold_train_metrics
    rep_val_metrics[[r]] <- fold_val_metrics
  }

  trial_train_metrics <- round(t(as.data.frame(rep_train_metrics)), 4)
  trial_val_metrics <- round(t(as.data.frame(rep_val_metrics)), 4)

  train_metrics <- colMeans(trial_train_metrics)
  val_metrics <- colMeans(trial_val_metrics)

  cv_structure <- expand.grid(folds=paste0("fold_", 1:folds), reps=paste0("rep_", 1:reps))
  trial_train_metrics <- cbind(cv_structure, trial_train_metrics)
  rownames(trial_train_metrics) <- NULL
  trial_val_metrics <- cbind(cv_structure, trial_val_metrics)
  rownames(trial_val_metrics) <- NULL

  ###TESTING STATS###LIMITED HOLDOUT TO CONTROL OVERFITTING
  history <- model %>% fit(x_train, y_train, epochs = epochs, batch_size=batch_size, verbose = verbose,
                           validation_data = list(x_test, y_test), callbacks = list(callback_early_stopping(monitor="val_loss",
                                                                                                            min_delta=min_delta, patience=floor(epochs*span),restore_best_weights=TRUE)))

  test_prediction <- pred_fun(x_test)
  reference <- y_orig[test_index,,drop=FALSE]
  if(task == "classif"){test_predicted <- test_prediction[, "prediction_class", drop = FALSE]; test_probs <- test_prediction[, 1:class_num, drop = FALSE]} else {test_predicted <- test_prediction}
  testing_frame <- as.data.frame(map2(reference, test_predicted, ~ data.frame(reference=.x, predicted=.y)))
  colnames(testing_frame) <- unlist(transpose(map(c("reference_", "predicted_"), ~ paste0(.x, target_names))))
  test_metrics <- suppressWarnings(eval_metrics(reference, test_predicted, task, positive))

  trials <- list(trial_train_metrics = trial_train_metrics, trial_val_metrics = trial_val_metrics)

  metrics <- round(rbind(train_metrics, val_metrics, test_metrics), 4)
  rownames(metrics) <- c("train", "valid", "test")

  toc(log = TRUE)
  time_log<-paste0(round(parse_number(unlist(tic.log()))/60, 0)," minutes")

  history_fixed <- history
  history_fixed$metrics <- map(history_fixed$metrics, ~ c(.x, rep(NA, epochs - length(.x))))
  plot <- plot(history_fixed)

  ###COLLECTED RESULTS
  outcome<-list(task = task, configuration = configuration, model = model, pred_fun = pred_fun,
                plot = plot, testing_frame = testing_frame, trials = trials,
                metrics = metrics, selected_feat = selected_feat, selected_inst =  selected_inst, time_log = time_log)

  tf$compat$v1$Session$close(sess)
  tf$keras$backend$clear_session

  return(outcome)
}
