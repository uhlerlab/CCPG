The implementation of FCI (stable) and RFCI (stable) uses the R package: pcalg (https://cran.r-project.org/web/packages/pcalg). 

In particular, we use the faster version with C++ (`skel.method=stable.fast`) for these methods in the package.

The benchmarks against other methods are done by running these methods on the same samples saved when running the other methods.

All R code can be found in this folder. 