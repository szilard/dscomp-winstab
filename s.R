library(data.table)
library(lightgbm)
library(ROCR)

N <- 100e3   ## train
V <-  20e3   ## early stopping valid
M <- 10e3    ## test ~ private LB
MM <- M*50   ## population
B <- 50      ## resamples

K <- 3000    ## competing models

set.seed(123)

d0 <- fread("train-10m.csv")
d0$dep_delayed_15min <- ifelse(d0$dep_delayed_15min=="Y",1,0)
N0 <- nrow(d0)

d0_wrules <- lgb.prepare_rules(d0)         # lightgbm cats
d0 <- d0_wrules$data
cols_cats <- names(d0_wrules$rules) 


idx_train <- sample(1:N0,N)
idx_valid <- sample(setdiff(1:N0,idx_train),V)
idx_popul <- sample(setdiff(setdiff(1:N0,idx_train),idx_valid),MM)

d_train <- d0[idx_train]
d_valid <- d0[idx_valid]
d_popul <- d0[idx_popul]

d_test <- list()
for (b in 1:B) {
  d_test[[b]] <- d_popul[sample(1:MM,M)]
}

p <- ncol(d0)-1
dlgb_train <- lgb.Dataset(data = as.matrix(d_train[,1:p]), label = d_train$dep_delayed_15min)
dlgb_valid <- lgb.Dataset(data = as.matrix(d_valid[,1:p]), label = d_valid$dep_delayed_15min)

rm(d0,d0_wrules)
gc()


params_grid <- expand.grid(                    
  num_leaves = c(1000,2000,4000,6000,8000,10000,12000), 
  learning_rate = c(0.03),          
  min_data_in_leaf = c(5,10,15,20,30),           
  feature_fraction = c(0.8),
  feature_fraction_bynode = c(0.8, 1),
  bagging_fraction = c(0.8),
  bagging_freq = c(0, 1),
  lambda_l1 = c(0, 0.01,0.03,0.1,0.3),  
  lambda_l2 = c(0, 0.01,0.03,0.1,0.3)
)
params_random <- params_grid[sample(1:nrow(params_grid),K),]


phat <- numeric(M)
auc_popul <- numeric(K)
auc_test <- matrix(0,nrow = K,ncol = B)
runtm_train <- numeric(K) 
ntrees <- numeric(K) 

auc <- function(md, df){
  phat <- predict(md, data = as.matrix(df[,1:p]))            
  rocr_pred <- prediction(phat, df$dep_delayed_15min)
  performance(rocr_pred, "auc")@y.values[[1]]
}

for (k in 1:K) {
  cat("***\n",k,"\n")
  params <- as.list(params_random[k,])

  runtm_train[k] <- system.time({
  md <- lgb.train(data = dlgb_train, objective = "binary",
              params = params,
              nrounds = 10000, early_stopping_rounds = 10, valid = list(valid = dlgb_valid), 
              num_threads = parallel::detectCores()/2,
              categorical_feature = cols_cats, 
              verbose = -1) 
  })[[3]]
  cat("Train runtime:",runtm_train[k],"\n")
  ntrees[k] <- md$best_iter

  runtm_popul <- system.time({
  auc_popul[k] <- auc(md, d_popul)
  })[[3]]
  cat("Popul runtime:",runtm_popul,"\n")

  runtm_test <- system.time({
  for (b in 1:B) {
    auc_test[k,b] <- auc(md, d_test[[b]])
  }
  })[[3]]
  cat("Test (bs) runtime:",runtm_test,"\n")

  if( k%%100==0 ) {
    save(k,auc_test,auc_popul,params_random,runtm_train,ntrees, file="simul-res.Rdata")
  }
}





