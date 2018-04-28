library(TUWmodel)

for(i in 1:20)
{
    # Open data
    filename <- file.path('input_data', 
                    sprintf('input_data_%0.2d.csv', i))
    data <- read.csv(filename, skip=17, stringsAsFactors=FALSE)
    names(data) <- c('DatesR', 'Precip', 'PotEvap', 'Runoff')

    # date conversion
    data$DatesR = as.POSIXlt(data$DatesR, format='%Y-%m-%d')

    # Parameters for snow melt
    SCF = 1.
    DDF = 1.
    Tr = 1.
    Ts = 0.
    Tm = 0.

    # Generate parameters - soil moisture/response func
    LPrat 
    FC
    BETA
    k0
    k1
    k2
    lsuz
    cperc
    bmax
    croute

    # prepare data
    params = c(SCF, DDF, Tr, Ts, Tm, 
                LPrat, FC, BETA,
                k0, k1, k2,
                lsuz, cperc, bmax, 
                croute)

    incon =c(50,0,2.5,2.5)

    # Run model - set air temperature to 30 deg C
    sim = TUWmodel(prec=data$Precip,
        airt=0*data$Precip + 30,
        ep=data$PotEvap,
        area=1,
        param=params,
        incon=incon)


    # Write data
    nm = names(outputs)[2:21]
    df = as.data.frame(outputs[nm])
    df$Qobs <- data$Runoff[indrun]
    filename <- file.path('output_data', 
                    sprintf('HBV_timeseries_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)

    df = as.data.frame(gr6j_params)
    filename <- file.path('output_data', 
                    sprintf('HBV_params_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)
}


