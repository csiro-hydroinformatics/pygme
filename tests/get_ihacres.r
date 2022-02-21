library(hydromad)

for(i in 1:20)
{
    # Open data
    filename <- file.path('input_data', 
                    sprintf('input_data_%0.2d.csv', i))
    data <- read.csv(filename, skip=17, stringsAsFactors=FALSE)
    names(data) <- c('DatesR', 'P', 'E', 'Qobs')

    filename <- file.path('input_data', 
                    sprintf('input_data_monthly_%0.2d.csv', i))
    datam <- read.csv(filename, skip=17, stringsAsFactors=FALSE)
    names(datam) <- c('DatesR', 'P', 'E', 'Qobs')

    # date conversion
    data$DatesR = as.POSIXlt(data$DatesR, format='%Y-%m-%d')
    datam$DatesR = as.POSIXlt(datam$DatesR, format='%Y-%m-%d')

    # ------ IHACRES ------------------
    mod <- hydromad(datam, sma="cmd")
    d <- runif(1, 1, 200)
    f <- runif(1, 0.01, 1)
    e <- runif(1, 0.01, 1)
    mod <- update(mod, d=d, f=f, e=e)
    outputs <- predict(mod, return_state=TRUE)

    # Write data
    df = as.data.frame(outputs)
    df$Qobs <- data$Runoff
    df$P <- datam$P
    df$E <- datam$E
    filename <- file.path('ihacres', 
                    sprintf('IHACRES_timeseries_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)
    df = as.data.frame(list(d=d, f=f, e=e))
    filename <- file.path('ihacres', 
                    sprintf('IHACRES_params_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)
}


