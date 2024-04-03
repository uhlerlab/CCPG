library('pcalg')
library('rhdf5')

nnodes <- 10

for (r in 1:5){
  print("-----------------RERUN---------------------")
  # run pc
  for (i in 1:50){
    x <- h5read(sprintf('./simulate-data/nsample/samples_%d_%d.h5', i*100, r-1), 'samples')
    suffStat <- list(C = cor(x), n = nrow(x))
    R <- pc(suffStat, indepTest=gaussCItest, p=nnodes, alpha=0.001, skel.method='stable.fast')
    if (all(names(R@graph@edgeData) == c("1|9","2|9","3|9","4|9","5|9","6|9","7|9","8|9","10|9"))){
      print(sprintf("PC %d", i*100))
      break
    }
  }
  
  # run fci
  for (i in 1:50){
    x <- h5read(sprintf('./simulate-data/nsample/samples_%d_%d.h5', i*100, r-1), 'samples')
    suffStat <- list(C = cor(x), n = nrow(x))
    R <- fci(suffStat, indepTest=gaussCItest, p=nnodes, alpha=0.999, skel.method='stable.fast')
    if (all(R@amat + diag(10) == 1)){
      print(sprintf("FCI %d", i*100))
      break
    }
  } 
  
  # run rfci
  for (i in 1:50){
    x <- h5read(sprintf('./simulate-data/nsample/samples_%d_%d.h5', i*100, r-1), 'samples')
    suffStat <- list(C = cor(x), n = nrow(x))
    RR <- rfci(suffStat, indepTest=gaussCItest, p=nnodes, alpha=0.999, skel.method='stable.fast')
    if (all(RR@amat + diag(10) == 1)){
      print(sprintf("RFCI %d", i*100))
      break
    }
  }  
}