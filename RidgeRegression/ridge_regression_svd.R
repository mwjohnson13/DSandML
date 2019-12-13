### Ridge Regression using Singular Value Decomposition ###
### Note: intercept is also penalized if included ###


#--Function: "ridgeReg" -------------------------------------------------------------------------------#
# Purpose: Perform ridge reression using singular value decomposition (SVD)
# Input  : X = design matrix
#          Y = response vector
#          lambda = penalty parameter (default is 0, which reduces to ordinary least squares)
#          intercept = boolean for whether intercept is already included in design matrix
#                      (TRUE means intercept included in X; default is FALSE)
# Output : A list containing regression coefficients, predicted values, residuals, 
#          and estimated variance-covariance of regression coefficients
#------------------------------------------------------------------------------------------------------#

ridgeReg <- function(X,Y,lambda=0,intercept=FALSE){
  n <- length(Y[,1])
  p <- length(X[1,])
  
  # convert data class to matrix
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  
  # add intercept if needed
  if( !intercept ){
    intercept <- rep(1, length(Prostate[,1]))
    X <- cbind(intercept, X)
  }
  
  # Compute svd
  U <- svd(X)$u
  V <- svd(X)$v
  D <- svd(X)$d
  
  # compute rank of X
  k <- sum(diag(D) > 0)
  
  # regression coefficients
  bRidge <- V %*% solve( diag(D^2) + lambda*diag(rep(1,k)) ) %*% diag(D) %*% t(U) %*% Y
  
  # variance-covariance of estimated regression coefficients
  term1 <- as.numeric( t(Y) %*% Y)
  term2 <- as.numeric( t(Y) %*% U %*% diag(D) %*% 
                         solve( diag(D^2) + lambda*diag(rep(1,k)) ) %*%
                         diag(D) %*% t(U) %*% Y)
  sigma2Hat <- (term1-term2)/(n-p)
  var_bRidge <- sigma2Hat * V %*% solve( diag(D^2) + lambda*diag(rep(1,k)) ) %*%
    diag(D^2) %*% solve( diag(D^2) + lambda*diag(rep(1,k)) ) %*% t(V)
  
  # fitted values
  fittedVals <- as.vector(U %*% diag(D) %*% solve( diag(D^2) + lambda*diag(rep(1,k)) ) %*%
                            diag(D) %*% t(U) %*% Y)
  
  # residuals
  residuals <- as.vector(Y - fittedVals)
  
  return(list(bRidge, fittedVals, residuals, var_bRidge))
}


