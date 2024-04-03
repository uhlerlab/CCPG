library('pcalg')
library('rhdf5')


for (nnodes in c(20)){
  print(sprintf("--------------Nnodes: %d----------------", nnodes))
  
  pc_runtime <- c()
  fci_runtime <- c()
  rfci_runtime <- c()
  #fciPlus_runtime <- c()
  
  for (r in 1:5){
    print(sprintf("--------------Run: %d----------------", r))
    
    x <- h5read(sprintf('./simulate-data/runtime/samples_%d_%d.h5', nnodes, r-1), 'samples')
    suffStat <- list(C = cor(x), n = nrow(x))
    
    # run pc
    start_time <- Sys.time()
    R <- pc(suffStat, indepTest=gaussCItest, p=nnodes, alpha=0.001, skel.method='stable.fast')
    end_time <- Sys.time()
    pc_runtime <- c(pc_runtime, difftime(end_time, start_time, units='secs'))
    print(sprintf("pc: %f", difftime(end_time, start_time, units='secs')))
    
    # run rfci
    start_time <- Sys.time()
    R <- rfci(suffStat, indepTest=gaussCItest, p=nnodes, alpha=0.999, skel.method='stable.fast')
    end_time <- Sys.time()
    rfci_runtime <- c(rfci_runtime, difftime(end_time, start_time, units='secs'))
    print(sprintf("rfci: %f", difftime(end_time, start_time, units='secs')))
    
    # run fci
    start_time <- Sys.time()
    R <- fci(suffStat, indepTest=gaussCItest, p=nnodes, alpha=0.999, skel.method='stable.fast')
    end_time <- Sys.time()
    fci_runtime <- c(fci_runtime, difftime(end_time, start_time, units='secs'))
    print(sprintf("fci: %f", difftime(end_time, start_time, units='secs')))
    
    # run fciPlus
    #start_time <- Sys.time()
    #R <- fciPlus(suffStat, indepTest=gaussCItest, p=nnodes, alpha=0.999, verbose=FALSE)
    #end_time <- Sys.time()
    #fciPlus_runtime <- c(fciPlus_runtime, end_time - start_time)
  }
  
  print("--------------Summary----------------")
  print(sprintf("pc: %f, %f", mean(pc_runtime), sd(pc_runtime)))
  print(sprintf("fci: %f, %f", mean(fci_runtime), sd(fci_runtime)))
  print(sprintf("rfci: %f, %f", mean(rfci_runtime), sd(rfci_runtime)))
  #print(sprintf("fci+: %f, %f", mean(fciPlus_runtime), sd(fciPlus_runtime)))
}